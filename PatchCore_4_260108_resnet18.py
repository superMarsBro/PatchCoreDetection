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
    "batch_extract_size": 8,  # <--- 之前报错就是缺这个
    # === 特征库构建参数 ===
    # "coreset_ratio": 0.10,  # 采样率
    # "max_coreset_size": 80000,  # 库大小限制
    "coreset_ratio": 0.15,  # 采样率
    "max_coreset_size": 100000,  # 库大小限制
    # === 匹配参数 (ResNet18 专用) ===
    "n_neighbors": 3,
    "nprobe": 5,  # 搜索范围
    # === 图像增强 ===
    "clahe_clip_limit": 2.0,
    # === 阈值策略 (自适应 + 绝对底线) ===
    # 策略：如果某个区域的分数 > 平均值 + 3倍标准差，就是瑕疵
    "sigma_threshold": 3.0,
    # 绝对底线：ResNet18 的特征距离通常在 0.0 ~ 20.0 之间 (归一化后)
    # 无论自适应怎么算，如果分数低于这个值，绝对不是瑕疵 (防止好图误报)
    # 如果你发现漏报严重，把这个数调小 (比如 3.0)
    # 如果你发现误报严重，把这个数调大 (比如 8.0)
    "absolute_floor": 0.15,
    "min_defect_area": 40,  # 忽略太小的噪点
    "morph_kernel_size": 3,  # 去噪核大小
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

        # LAB 增强
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


# ===================== 2. 特征提取器 (ResNet18) =====================
class ResNetFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT
        base_model = resnet18(weights=weights)

        # 提取 layer1 (浅层细节) 和 layer2 (中层纹理)
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
        # 上采样对齐
        s2_up = torch.nn.functional.interpolate(
            outputs["s2"], size=outputs["s1"].shape[-2:], mode="bilinear"
        )
        features = torch.cat([outputs["s1"], s2_up], dim=1)
        features = self.avg_pool(features)

        self.feature_map_h, self.feature_map_w = features.shape[2], features.shape[3]
        B, C, H, W = features.shape
        return features.permute(0, 2, 3, 1).reshape(B, H * W, C)


extractor = ResNetFeatureExtractor().to(DEVICE)


# ===================== 3. 构建特征库 =====================
def build_dynamic_library(normal_dir, save_path):
    logging.info("🚀 [模式1] 构建库 - ResNet18 Auto-Clustering")
    img_paths = [
        os.path.join(r, f)
        for r, _, fs in os.walk(normal_dir)
        for f in fs
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]
    if not img_paths:
        logging.error("❌ 没找到图片")
        return

    # 1. 提取特征
    features_list = []
    # 内存保护机制
    TARGET_MEM = 1500000
    est_patches = len(img_paths) * (
        CONFIG["input_size"][0] // 4 * CONFIG["input_size"][1] // 4
    )
    ratio = min(1.0, TARGET_MEM / est_patches)
    logging.info(f"📊 动态采样率: {ratio:.4f}")

    extractor.eval()
    # 这里用到了 batch_extract_size，确保 CONFIG 里有
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

        # 采样
        for f in feat:
            n = f.shape[0]
            if ratio < 1.0:
                idx = np.random.choice(n, int(n * ratio), replace=False)
                features_list.append(f[idx])
            else:
                features_list.append(f)

    all_features = np.concatenate(features_list, axis=0)

    # 归一化 (关键)
    all_features = all_features / (
        np.linalg.norm(all_features, axis=1, keepdims=True) + 1e-8
    )

    # 2. Coreset 压缩
    if all_features.shape[0] > CONFIG["max_coreset_size"]:
        logging.info("⏳ Coreset压缩中...")
        from sklearn.cluster import MiniBatchKMeans

        n_clusters = CONFIG["max_coreset_size"] // 10
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters, batch_size=4096, n_init="auto"
        ).fit(all_features)
        core_features = kmeans.cluster_centers_
    else:
        core_features = all_features

    logging.info(f"✅ 核心库大小: {core_features.shape}")

    # 3. 构建索引
    if FAISS_AVAILABLE:
        d = core_features.shape[1]
        # 自动计算聚类数
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

    def detect(self, img_path, save_dir):
        pre = robust_preprocess(img_path)
        if not pre:
            return

        extractor.eval()
        with torch.no_grad():
            feat = extractor(pre["tensor"].to(DEVICE))
            feat = feat.cpu().numpy().astype(np.float32)[0]
            fh, fw = extractor.feature_map_h, extractor.feature_map_w

        # === L2 归一化 (必须做) ===
        norm = np.linalg.norm(feat, axis=1, keepdims=True) + 1e-8
        feat = feat / norm

        distances, _ = self.index.search(feat, CONFIG["n_neighbors"])

        # 距离平均
        score_map = np.mean(distances, axis=1).reshape(fh, fw)
        score_map = cv2.GaussianBlur(score_map, (3, 3), 0)

        h, w = CONFIG["input_size"]
        heatmap = cv2.resize(score_map, (w, h), interpolation=cv2.INTER_LINEAR)

        self.save_result(img_path, pre["processed_img"], heatmap, save_dir)

    def save_result(self, path, img, heatmap, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        name = os.path.splitext(os.path.basename(path))[0]

        # 统计数据
        h_min, h_mean, h_max = np.min(heatmap), np.mean(heatmap), np.max(heatmap)
        h_std = np.std(heatmap)

        # === 自适应阈值计算 ===
        # 阈值 = 平均值 + N * 标准差
        dynamic_thresh = h_mean + (CONFIG["sigma_threshold"] * h_std)
        # 取较大值作为最终阈值
        final_thresh = max(dynamic_thresh, CONFIG["absolute_floor"])

        logging.info(
            f"🖼️ {name} | Max:{h_max:.2f} | Thresh:{final_thresh:.2f} (Abs:{CONFIG['absolute_floor']})"
        )

        mask = (heatmap > final_thresh).astype(np.uint8) * 255

        # 去噪
        k = CONFIG["morph_kernel_size"]
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        res_img = img.copy()
        count = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > CONFIG["min_defect_area"]:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(res_img, (x, y), (x + w, y + h), (0, 0, 255), 2)

                # 计算分数
                roi_mask = np.zeros_like(heatmap, dtype=np.uint8)
                cv2.drawContours(roi_mask, [cnt], -1, 1, -1)
                mean_val = cv2.mean(heatmap, mask=roi_mask)[0]

                label = f"{mean_val:.1f}"
                cv2.putText(
                    res_img,
                    label,
                    (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                )
                count += 1

        if count > 0:
            logging.info(f"❌ NG: {name} ({count} 处)")
            cv2.imwrite(os.path.join(save_dir, f"{name}_NG.jpg"), res_img)

            # 可视化热力图 (自动拉伸对比度，方便人眼看)
            hm_norm = (heatmap - h_min) / (h_max - h_min + 1e-8)
            hm_color = cv2.applyColorMap(
                (hm_norm * 255).astype(np.uint8), cv2.COLORMAP_JET
            )
            cv2.imwrite(os.path.join(save_dir, f"{name}_heat.jpg"), hm_color)
        else:
            logging.info(f"✅ OK: {name}")


# ===================== 主程序 =====================
def main():
    print("🚀 智能布料检测系统 V5 (ResNet18 + 自适应阈值)")

    # === 请在这里修改你的路径 ===
    DATASET_DIR = r"F:\\All_dataset\\\\long_buliao_datasets_260107_cutbackground"
    LIB_FILE = "PatchCoreDetection\\\\feature_lib\\\\lib_resnet18_260109_1.npz"
    TEST_DIR = r"F:\All_dataset\\half_bad_half_good"

    action = input("1: 构建新库, 2: 检测\n选择: ")
    if action == "1":
        build_dynamic_library(DATASET_DIR, LIB_FILE)
    elif action == "2":
        if not os.path.exists(LIB_FILE):
            print("❌ 请先运行模式1")
            return

        detector = AutoDefectDetector(LIB_FILE)

        valid_exts = (".jpg", ".jpeg", ".png", ".bmp")
        imgs = [
            os.path.join(r, f)
            for r, _, fs in os.walk(TEST_DIR)
            for f in fs
            if f.lower().endswith(valid_exts)
        ]

        print(f"📊 待检测图片: {len(imgs)} 张")
        for p in tqdm(imgs):
            detector.detect(p, "results_v5_auto")


if __name__ == "__main__":
    main()
