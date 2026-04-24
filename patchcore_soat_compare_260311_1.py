import torch
import torch.nn as nn
import cv2
import numpy as np
import os
import warnings
import json
import time
import math
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models.feature_extraction import create_feature_extractor
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix

# ===================== 日志配置 =====================
file_handler = logging.FileHandler(
    "defect_detection_v5.log", mode="w", encoding="utf-8"
)
stream_handler = logging.StreamHandler()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[file_handler, stream_handler],
)

# ===================== 环境检查 =====================
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("⚠️ 未安装 FAISS，速度将受限")

# ===================== 全局配置 =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIG = {
    # === 基础参数 ===
    "input_size": (384, 1024),
    "batch_extract_size": 8,
    # === 特征库构建参数 ===
    "coreset_ratio": 0.20,  # 采样率
    "max_coreset_size": 100000,  # 库大小限制
    # === 匹配参数 ===
    "n_neighbors": 9,
    "nprobe": 5,
    # === 图像增强 ===
    "clahe_clip_limit": 1.0,
    # === 阈值策略 ===
    "sigma_threshold": 3.0,
    # 【待校准参数】
    "absolute_floor": 0.20,
    "min_defect_area": 40,
    "morph_kernel_size": 3,
}

# ===================== 1. 预处理 =====================
def robust_preprocess(img_path):
    try:
        img_original = cv2.imread(img_path)
        if img_original is None:
            return None
        target_h, target_w = CONFIG["input_size"]
        img_resized = cv2.resize(
            img_original, (target_w, target_h), interpolation=cv2.INTER_LINEAR
        )
        lab = cv2.cvtColor(img_resized, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(
            clipLimit=CONFIG["clahe_clip_limit"], tileGridSize=(8, 8)
        )
        l = clahe.apply(l)
        img_enhanced = cv2.merge((l, a, b))
        img_enhanced = cv2.cvtColor(img_enhanced, cv2.COLOR_LAB2BGR)

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        img_rgb = cv2.cvtColor(img_enhanced, cv2.COLOR_BGR2RGB)
        img_tensor = transform(img_rgb).unsqueeze(0)
        return {"tensor": img_tensor, "processed_img": img_enhanced}
    except Exception as e:
        return None

# ===================== 2. 特征提取器 (优化版) =====================
class ResNetFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT
        base_model = resnet18(weights=weights)
        self.feature_extractor = create_feature_extractor(
            base_model, return_nodes={"layer2": "s2", "layer3": "s3"}
        )
        self.avg_pool = nn.AvgPool2d(5, stride=1, padding=2)
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.feature_extractor.eval()
        self.feature_map_h = 0
        self.feature_map_w = 0

    def forward(self, x):
        outputs = self.feature_extractor(x)
        s3_up = torch.nn.functional.interpolate(
            outputs["s3"], size=outputs["s2"].shape[-2:], mode="bilinear"
        )
        features = torch.cat([outputs["s2"], s3_up], dim=1)
        features = self.avg_pool(features)
        self.feature_map_h, self.feature_map_w = features.shape[2], features.shape[3]
        B, C, H, W = features.shape
        return features.permute(0, 2, 3, 1).reshape(B, H * W, C)

extractor = ResNetFeatureExtractor().to(DEVICE)

# ===================== 3. 构建特征库 =====================
def build_dynamic_library(normal_dir, save_path):
    logging.info("🚀 [模式1] 构建特征库...")
    img_paths = [
        os.path.join(r, f)
        for r, _, fs in os.walk(normal_dir)
        for f in fs
        if f.lower().endswith((".jpg", ".png"))
    ]
    if not img_paths:
        logging.error("❌ 没找到图片")
        return

    features_list = []
    TARGET_MEM = 1500000
    est_patches = len(img_paths) * (
        CONFIG["input_size"][0] // 4 * CONFIG["input_size"][1] // 4
    )
    ratio = min(1.0, TARGET_MEM / est_patches)
    logging.info(f"📊 动态采样率: {ratio:.4f}")

    extractor.eval()
    for i in tqdm(range(0, len(img_paths), CONFIG["batch_extract_size"])):
        batch = []
        for p in img_paths[i : i + CONFIG["batch_extract_size"]]:
            res = robust_preprocess(p)
            if res:
                batch.append(res["tensor"])
        if not batch:
            continue

        with torch.no_grad():
            feat = extractor(torch.cat(batch).to(DEVICE)).cpu().numpy()

        for f in feat:
            n = f.shape[0]
            if ratio < 1.0:
                idx = np.random.choice(n, int(n * ratio), replace=False)
                features_list.append(f[idx])
            else:
                features_list.append(f)

    all_features = np.concatenate(features_list, axis=0)
    all_features = all_features / (
        np.linalg.norm(all_features, axis=1, keepdims=True) + 1e-8
    )

    if all_features.shape[0] > CONFIG["max_coreset_size"]:
        logging.info("⏳ 聚类压缩中...")
        from sklearn.cluster import MiniBatchKMeans

        kmeans = MiniBatchKMeans(
            n_clusters=CONFIG["max_coreset_size"] // 10, batch_size=4096, n_init="auto"
        ).fit(all_features)
        core_features = kmeans.cluster_centers_
    else:
        core_features = all_features

    logging.info(f"✅ 核心库大小: {core_features.shape}")

    if FAISS_AVAILABLE:
        d = core_features.shape[1]
        nlist = int(8 * math.sqrt(core_features.shape[0]))
        nlist = max(32, min(nlist, 2048))
        logging.info(f"🧠 构建索引: {nlist} 类")
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
        index.train(core_features)
        index.add(core_features)
        faiss.write_index(index, save_path.replace(".npz", ".ivf_index"))

    np.savez(save_path, features=core_features)

# ===================== 4. 检测器 =====================
class AutoDefectDetector:
    def __init__(self, lib_path):
        self.lib_path = lib_path
        self.load_index()

    def load_index(self):
        idx_path = self.lib_path.replace(".npz", ".ivf_index")
        self.index = faiss.read_index(idx_path)
        self.index.nprobe = CONFIG["nprobe"]
        if DEVICE.type == "cuda" and hasattr(faiss, "StandardGpuResources"):
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
            except:
                pass

    def get_heatmap(self, img_path):
        pre = robust_preprocess(img_path)
        if not pre:
            return None, None

        extractor.eval()
        with torch.no_grad():
            feat = extractor(pre["tensor"].to(DEVICE))
            feat = feat.cpu().numpy().astype(np.float32)[0]
            fh, fw = extractor.feature_map_h, extractor.feature_map_w

        norm = np.linalg.norm(feat, axis=1, keepdims=True) + 1e-8
        feat = feat / norm

        distances, _ = self.index.search(feat, CONFIG["n_neighbors"])
        score_map = np.mean(distances, axis=1).reshape(fh, fw)
        score_map = cv2.GaussianBlur(score_map, (9, 9), 0)

        h, w = CONFIG["input_size"]
        heatmap = cv2.resize(score_map, (w, h), interpolation=cv2.INTER_LINEAR)
        return heatmap, pre["processed_img"]

    def compute_score_only(self, img_path):
        heatmap, _ = self.get_heatmap(img_path)
        if heatmap is None:
            return 0.0
        return np.max(heatmap)

    def detect_and_save(self, img_path, save_dir):
        heatmap, img = self.get_heatmap(img_path)
        if heatmap is None:
            return

        name = os.path.splitext(os.path.basename(img_path))[0]
        h_mean, h_std = np.mean(heatmap), np.std(heatmap)
        h_max = np.max(heatmap)

        dynamic_thresh = h_mean + (CONFIG["sigma_threshold"] * h_std)
        final_thresh = max(dynamic_thresh, CONFIG["absolute_floor"])

        logging.info(f"🖼️ {name} | Max: {h_max:.4f} | Limit: {final_thresh:.4f}")

        if h_max > final_thresh:
            mask = (heatmap > final_thresh).astype(np.uint8) * 255
            k = CONFIG["morph_kernel_size"]
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            res_img = img.copy()
            count = 0
            for cnt in contours:
                if cv2.contourArea(cnt) > CONFIG["min_defect_area"]:
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(res_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    count += 1

            if count > 0:
                logging.info(f"❌ NG: {name} ({count} 处)")
                cv2.imwrite(os.path.join(save_dir, f"{name}_NG.jpg"), res_img)
                hm_norm = (heatmap - heatmap.min()) / (
                    heatmap.max() - heatmap.min() + 1e-8
                )
                hm_color = cv2.applyColorMap(
                    (hm_norm * 255).astype(np.uint8), cv2.COLORMAP_JET
                )
                cv2.imwrite(os.path.join(save_dir, f"{name}_heat.jpg"), hm_color)
                return

        logging.info(f"✅ OK: {name}")

# ===================== 5. 为对比实验添加的模型实现 =====================
# === 原版PatchCore (使用layer1和layer2，平滑核3) ===
class OriginalPatchCoreFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT
        base_model = resnet18(weights=weights)
        self.feature_extractor = create_feature_extractor(
            base_model, return_nodes={"layer1": "s1", "layer2": "s2"}
        )
        self.avg_pool = nn.AvgPool2d(3, stride=1, padding=1)
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.feature_extractor.eval()
        self.feature_map_h = 0
        self.feature_map_w = 0

    def forward(self, x):
        outputs = self.feature_extractor(x)
        s2_up = torch.nn.functional.interpolate(
            outputs["s2"], size=outputs["s1"].shape[-2:], mode="bilinear"
        )
        features = torch.cat([outputs["s1"], s2_up], dim=1)
        features = self.avg_pool(features)
        self.feature_map_h, self.feature_map_w = features.shape[2], features.shape[3]
        B, C, H, W = features.shape
        return features.permute(0, 2, 3, 1).reshape(B, H * W, C)

class OriginalPatchCoreDetector(AutoDefectDetector):
    def __init__(self, lib_path):
        super().__init__(lib_path)
        self.feature_extractor = OriginalPatchCoreFeatureExtractor().to(DEVICE)
    
    def get_heatmap(self, img_path):
        pre = robust_preprocess(img_path)
        if not pre:
            return None, None

        self.feature_extractor.eval()
        with torch.no_grad():
            feat = self.feature_extractor(pre["tensor"].to(DEVICE))
            feat = feat.cpu().numpy().astype(np.float32)[0]
            fh, fw = self.feature_extractor.feature_map_h, self.feature_extractor.feature_map_w

        norm = np.linalg.norm(feat, axis=1, keepdims=True) + 1e-8
        feat = feat / norm

        distances, _ = self.index.search(feat, CONFIG["n_neighbors"])
        score_map = np.mean(distances, axis=1).reshape(fh, fw)
        score_map = cv2.GaussianBlur(score_map, (3, 3), 0)  # 平滑核为3

        h, w = CONFIG["input_size"]
        heatmap = cv2.resize(score_map, (w, h), interpolation=cv2.INTER_LINEAR)
        return heatmap, pre["processed_img"]

# === PaDiM (使用layer3特征，不使用平滑) ===
class PaDiMFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT
        base_model = resnet18(weights=weights)
        self.feature_extractor = create_feature_extractor(
            base_model, return_nodes={"layer3": "s3"}
        )
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.feature_extractor.eval()
        self.feature_map_h = 0
        self.feature_map_w = 0

    def forward(self, x):
        outputs = self.feature_extractor(x)
        features = outputs["s3"]
        self.feature_map_h, self.feature_map_w = features.shape[2], features.shape[3]
        B, C, H, W = features.shape
        return features.permute(0, 2, 3, 1).reshape(B, H * W, C)

class PaDiMDetector(AutoDefectDetector):
    def __init__(self, lib_path):
        super().__init__(lib_path)
        self.feature_extractor = PaDiMFeatureExtractor().to(DEVICE)
    
    def get_heatmap(self, img_path):
        pre = robust_preprocess(img_path)
        if not pre:
            return None, None

        self.feature_extractor.eval()
        with torch.no_grad():
            feat = self.feature_extractor(pre["tensor"].to(DEVICE))
            feat = feat.cpu().numpy().astype(np.float32)[0]
            fh, fw = self.feature_extractor.feature_map_h, self.feature_extractor.feature_map_w

        norm = np.linalg.norm(feat, axis=1, keepdims=True) + 1e-8
        feat = feat / norm

        distances, _ = self.index.search(feat, CONFIG["n_neighbors"])
        score_map = np.mean(distances, axis=1).reshape(fh, fw)
        # PaDiM不使用高斯模糊，保持原始热力图

        h, w = CONFIG["input_size"]
        heatmap = cv2.resize(score_map, (w, h), interpolation=cv2.INTER_LINEAR)
        return heatmap, pre["processed_img"]

# === MADNet (多尺度特征融合) ===
class MADNetFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT
        base_model = resnet18(weights=weights)
        self.feature_extractor = create_feature_extractor(
            base_model, 
            return_nodes={
                "layer1": "s1", 
                "layer2": "s2", 
                "layer3": "s3"
            }
        )
        self.avg_pool = nn.AvgPool2d(5, stride=1, padding=2)
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.feature_extractor.eval()
        self.feature_map_h = 0
        self.feature_map_w = 0

    def forward(self, x):
        outputs = self.feature_extractor(x)
        
        # 将s3上采样到s2尺寸
        s3_up = torch.nn.functional.interpolate(
            outputs["s3"], size=outputs["s2"].shape[-2:], mode="bilinear"
        )
        # 将s2上采样到s1尺寸
        s2_up = torch.nn.functional.interpolate(
            outputs["s2"], size=outputs["s1"].shape[-2:], mode="bilinear"
        )
        
        # 拼接所有特征
        features = torch.cat([
            outputs["s1"], 
            s2_up, 
            s3_up
        ], dim=1)
        
        features = self.avg_pool(features)
        self.feature_map_h, self.feature_map_w = features.shape[2], features.shape[3]
        B, C, H, W = features.shape
        return features.permute(0, 2, 3, 1).reshape(B, H * W, C)

class MADNetDetector(AutoDefectDetector):
    def __init__(self, lib_path):
        super().__init__(lib_path)
        self.feature_extractor = MADNetFeatureExtractor().to(DEVICE)
    
    def get_heatmap(self, img_path):
        pre = robust_preprocess(img_path)
        if not pre:
            return None, None

        self.feature_extractor.eval()
        with torch.no_grad():
            feat = self.feature_extractor(pre["tensor"].to(DEVICE))
            feat = feat.cpu().numpy().astype(np.float32)[0]
            fh, fw = self.feature_extractor.feature_map_h, self.feature_extractor.feature_map_w

        norm = np.linalg.norm(feat, axis=1, keepdims=True) + 1e-8
        feat = feat / norm

        distances, _ = self.index.search(feat, CONFIG["n_neighbors"])
        score_map = np.mean(distances, axis=1).reshape(fh, fw)
        score_map = cv2.GaussianBlur(score_map, (9, 9), 0)

        h, w = CONFIG["input_size"]
        heatmap = cv2.resize(score_map, (w, h), interpolation=cv2.INTER_LINEAR)
        return heatmap, pre["processed_img"]

# ===================== 6. 学术评估函数 (已修改以支持返回结果) =====================
def evaluate_academic_metrics(detector, good_dir, bad_dir, return_results=False):
    print("\n" + "=" * 50)
    print("🔬 开始学术性能评估 (Image-level Metrics)")
    print("=" * 50)

    good_imgs = [
        os.path.join(good_dir, f)
        for f in os.listdir(good_dir)
        if f.endswith((".jpg", ".png"))
    ]
    y_true_good = [0] * len(good_imgs)

    bad_imgs = [
        os.path.join(bad_dir, f)
        for f in os.listdir(bad_dir)
        if f.endswith((".jpg", ".png"))
    ]
    y_true_bad = [1] * len(bad_imgs)

    all_imgs = good_imgs + bad_imgs
    y_true = np.array(y_true_good + y_true_bad)
    y_scores = []
    y_preds = []
    max_good_score = 0.0

    if len(all_imgs) == 0:
        print("❌ 未找到测试图片，请检查目录。")
        return

    print(
        f"📊 测试集样本: 正常(Good)={len(good_imgs)} 张, 瑕疵(Bad)={len(bad_imgs)} 张"
    )

    for img_path, label in tqdm(
        zip(all_imgs, y_true), total=len(all_imgs), desc="评估中"
    ):
        heatmap, _ = detector.get_heatmap(img_path)
        if heatmap is None:
            y_scores.append(0.0)
            y_preds.append(0)
            continue

        h_max = np.max(heatmap)
        h_mean = np.mean(heatmap)
        h_std = np.std(heatmap)

        y_scores.append(h_max)

        if label == 0 and h_max > max_good_score:
            max_good_score = h_max  # 收集正常样本里的最高分

        dynamic_thresh = h_mean + (CONFIG["sigma_threshold"] * h_std)
        final_thresh = max(dynamic_thresh, CONFIG["absolute_floor"])

        if h_max > final_thresh:
            y_preds.append(1)
        else:
            y_preds.append(0)

    y_scores = np.array(y_scores)
    y_preds = np.array(y_preds)

    auroc = roc_auc_score(y_true, y_scores)
    tn, fp, fn, tp = confusion_matrix(y_true, y_preds).ravel()
    fpr = fp / (fp + tn + 1e-8)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_preds, average="binary", zero_division=0
    )

    print("\n" + "🌟 学术论文评估指标结果 (Results for Paper) 🌟")
    print("-" * 50)
    print(f"   ➤ Image-level AUROC : {auroc * 100:.2f} %")
    print(f"   ➤ MaxScore on Good  : {max_good_score:.4f} (越低越好)")
    print("-" * 50)
    print(f"   ➤ Precision (查准率) : {precision * 100:.2f} %")
    print(f"   ➤ Recall (查全/召回) : {recall * 100:.2f} %")
    print(f"   ➤ F1-Score (综合)    : {f1 * 100:.2f} %")
    print(f"   ➤ FPR (误检/假阳率)  : {fpr * 100:.2f} %  ({fp} 张好布被误判)")
    print("=" * 50)

    # 绘制 ROC
    fpr_curve, tpr_curve, _ = roc_curve(y_true, y_scores)
    plt.figure(figsize=(8, 6))
    plt.plot(
        fpr_curve, tpr_curve, color="red", lw=2, label=f"Ours (AUROC = {auroc:.4f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig("ROC_Curve_Ablation.png", dpi=300)

    if return_results:
        return auroc, f1, recall, max_good_score, fpr

    return auroc

# ===================== 7. 对比实验功能 =====================
def compare_sota_models(detectors, good_dir, bad_dir, num_runs=3):
    """
    进行SOTA模型对比实验
    detectors: 字典，包含模型名到AutoDefectDetector实例的映射
    num_runs: 重复运行次数，用于计算平均值
    """
    results = {}
    
    # 为每个模型准备结果存储
    for model_name in detectors:
        results[model_name] = {
            "auroc": [],
            "fps": [],
            "memory_usage": [],
            "f1": [],
            "recall": []
        }
    
    # 为每个模型运行多次实验
    for run in range(num_runs):
        print(f"\n{'='*50}")
        print(f"Run {run+1}/{num_runs} - SOTA Model Comparison")
        print(f"{'='*50}")
        
        for model_name, detector in detectors.items():
            print(f"\n📊 评估模型: {model_name}")
            
            # 评估AUROC
            auroc = evaluate_academic_metrics(detector, good_dir, bad_dir)
            
            # 评估推理帧率
            fps = measure_inference_fps(detector, good_dir, bad_dir)
            
            # 评估内存占用
            memory_usage = measure_memory_usage(detector)
            
            # 重新评估F1和Recall
            _, f1, recall, _, _ = evaluate_academic_metrics(
                detector, good_dir, bad_dir, return_results=True
            )
            
            # 保存结果
            results[model_name]["auroc"].append(auroc)
            results[model_name]["fps"].append(fps)
            results[model_name]["memory_usage"].append(memory_usage)
            results[model_name]["f1"].append(f1)
            results[model_name]["recall"].append(recall)
    
    # 计算平均结果
    for model_name in results:
        for metric in results[model_name]:
            results[model_name][metric] = np.mean(results[model_name][metric])
    
    # 打印结果表格
    print("\n" + "="*50)
    print("📊 SOTA 模型对比结果 (平均值)")
    print("="*50)
    print(f"{'Model':<25} | {'AUROC (%)':<15} | {'FPS':<10} | {'Memory (MB)':<15} | {'F1 (%)':<10} | {'Recall (%)':<10}")
    print("-"*100)
    
    for model_name, metrics in results.items():
        print(f"{model_name:<25} | {metrics['auroc']*100:<15.2f} | {metrics['fps']:<10.2f} | {metrics['memory_usage']:<15.2f} | {metrics['f1']*100:<10.2f} | {metrics['recall']*100:<10.2f}")
    
    # 保存结果到文件
    with open("sota_comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # 生成可视化图表
    generate_comparison_plots(results)
    
    return results

def measure_inference_fps(detector, good_dir, bad_dir, num_samples=100):
    """
    测量模型的推理帧率 (FPS)
    """
    start_time = time.time()
    
    # 选择一些样本进行推理
    all_imgs = [
        os.path.join(good_dir, f) for f in os.listdir(good_dir) if f.endswith((".jpg", ".png"))
    ] + [
        os.path.join(bad_dir, f) for f in os.listdir(bad_dir) if f.endswith((".jpg", ".png"))
    ]
    
    # 随机选择样本
    if len(all_imgs) > num_samples:
        import random
        all_imgs = random.sample(all_imgs, num_samples)
    
    # 进行推理
    for img_path in tqdm(all_imgs, desc="测量FPS"):
        detector.compute_score_only(img_path)
    
    total_time = time.time() - start_time
    fps = num_samples / total_time
    return fps

def measure_memory_usage(detector):
    """
    测量模型的GPU内存占用
    """
    # 重置CUDA统计
    torch.cuda.reset_peak_memory_stats()
    
    # 创建一个样本
    sample_img = os.path.join(EVAL_GOOD_DIR, os.listdir(EVAL_GOOD_DIR)[0])
    detector.compute_score_only(sample_img)
    
    # 获取峰值内存使用
    memory_usage = torch.cuda.max_memory_allocated() / 1024 / 1024  # 转换为MB
    return memory_usage

def generate_comparison_plots(results):
    """
    生成对比实验的可视化图表
    """
    import matplotlib.pyplot as plt
    
    # 创建AUROC和FPS的对比图
    models = list(results.keys())
    auroc_values = [results[model]["auroc"] * 100 for model in models]
    fps_values = [results[model]["fps"] for model in models]
    
    plt.figure(figsize=(10, 6))
    plt.bar(models, auroc_values, color='blue', alpha=0.7, label='AUROC (%)')
    plt.plot(models, fps_values, color='red', marker='o', linewidth=2, label='FPS')
    
    plt.xlabel('Model')
    plt.ylabel('Performance')
    plt.title('SOTA Model Comparison (AUROC & FPS)')
    plt.legend()
    plt.xticks(rotation=15)
    plt.grid(True, alpha=0.3)
    plt.savefig('sota_comparison_auroc_fps.png', dpi=300)
    
    # 创建F1和Recall的对比图
    f1_values = [results[model]["f1"] * 100 for model in models]
    recall_values = [results[model]["recall"] * 100 for model in models]
    
    plt.figure(figsize=(10, 6))
    plt.bar(models, f1_values, color='green', alpha=0.7, label='F1 Score (%)')
    plt.bar(models, recall_values, color='orange', alpha=0.7, label='Recall (%)')
    
    plt.xlabel('Model')
    plt.ylabel('Performance')
    plt.title('SOTA Model Comparison (F1 & Recall)')
    plt.legend()
    plt.xticks(rotation=15)
    plt.grid(True, alpha=0.3)
    plt.savefig('sota_comparison_f1_recall.png', dpi=300)
    
    print("\n📊 对比结果图表已保存: sota_comparison_auroc_fps.png, sota_comparison_f1_recall.png")

# ===================== 主程序 =====================
def main():
    print("🚀 智能布料检测系统 V5 (含学术评估模式)")

    # 您的训练特征库相关路径 (不需要改)
    DATASET_DIR = r"F:\All_dataset\long_buliao_datasets_260107_cutbackground"
    LIB_FILE = "feature_lib\lib_resnet18_260113_1.npz"
    CALIBRATION_DIR = r"F:\\All_dataset\\\\long_buliao_datasets_260107_cutbackground"

    # === 请在这里填入您刚才新建的【固定测试集】路径 ===
    EVAL_GOOD_DIR = r"F:\All_dataset\260309buliao_datasets\good"  # 纯正常图片
    EVAL_BAD_DIR = r"F:\All_dataset\260309buliao_datasets\bad"  # 纯瑕疵图片

    print("\n请选择模式:")
    print("1: 构建新库 (Training)")
    print("2: 批量检测 (Testing)")
    print("3: 阈值校准 (Calibration - 统计正常图的最大分数)")
    print("4: 评估学术指标 (Ablation Study 专用)")
    print("5: SOTA 模型对比实验 (对比不同模型性能)")

    action = input("选择: ")

    if action == "1":
        build_dynamic_library(DATASET_DIR, LIB_FILE)
    elif action == "2":
        if not os.path.exists(LIB_FILE):
            print("❌ 请先运行模式1")
            return
        SAVE_DIR = "results_v5_final"
        if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR)
            print(f"📂 已创建结果文件夹: {SAVE_DIR}")
        detector = AutoDefectDetector(LIB_FILE)
        TEST_DIR = r"F:\All_dataset\long_negative_buliao_260107_cutbackground"
        imgs = [
            os.path.join(r, f)
            for r, _, fs in os.walk(TEST_DIR)
            for f in fs
            if f.lower().endswith((".jpg", ".png"))
        ]
        print(f"📊 开始检测 {len(imgs)} 张图片...")
        for p in tqdm(imgs):
            detector.detect_and_save(p, SAVE_DIR)
    elif action == "3":
        if not os.path.exists(LIB_FILE):
            print("❌ 请先运行模式1")
            return
        detector = AutoDefectDetector(LIB_FILE)
        calib_imgs = [
            os.path.join(r, f)
            for r, _, fs in os.walk(CALIBRATION_DIR)
            for f in fs
            if f.lower().endswith((".jpg", ".png"))
        ]
        if len(calib_imgs) > 100:
            import random
            calib_imgs = random.sample(calib_imgs, 100)
            print(f"ℹ️ 图片太多，随机抽取 100 张进行统计...")
        print(f"📊 正在计算 {len(calib_imgs)} 张正常图片的分数分布...")
        max_scores = [detector.compute_score_only(p) for p in tqdm(calib_imgs)]
        avg_max = np.mean(max_scores)
        std_max = np.std(max_scores)
        global_max = np.max(max_scores)
        print("\n" + "=" * 40)
        print("📊 正常图片分数统计结果")
        print("=" * 40)
        print(f"图片数量: {len(max_scores)}")
        print(f"平均 MaxScore: {avg_max:.4f}")
        print(f"标准差 StdDev: {std_max:.4f}")
        print(f"⚠️ 全局最高分 (Worst Case): {global_max:.4f}")
        print("-" * 40)
        suggested_floor = global_max + 0.05
        print(f"✅ 建议设置 absolute_floor = {suggested_floor:.2f}")
        print("=" * 40 + "\n")
        print(f"请修改 CONFIG 中的 'absolute_floor' 为上述建议值，然后运行模式 2。")
    elif action == "4":
        if not os.path.exists(LIB_FILE):
            print("❌ 请先运行模式1建库")
            return
        detector = AutoDefectDetector(LIB_FILE)
        evaluate_academic_metrics(detector, EVAL_GOOD_DIR, EVAL_BAD_DIR)
    elif action == "5":
        if not os.path.exists(LIB_FILE):
            print("❌ 请先运行模式1建库")
            return
        
        # 创建不同模型的检测器
        detectors = {}
        
        # 原版PatchCore
        detectors["Original PatchCore"] = OriginalPatchCoreDetector(LIB_FILE)
        
        # 改进版PatchCore
        detectors["Improved PatchCore"] = AutoDefectDetector(LIB_FILE)
        
        # PaDiM
        PADIM_LIB_FILE = "feature