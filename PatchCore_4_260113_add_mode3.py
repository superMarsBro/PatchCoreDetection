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

# good example

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
    "coreset_ratio": 0.15,  # 采样率
    "max_coreset_size": 100000,  # 库大小限制
    # === 匹配参数 ===
    "n_neighbors": 3,
    "nprobe": 5,
    # === 图像增强 ===
    "clahe_clip_limit": 2.0,
    # === 阈值策略 ===
    "sigma_threshold": 3.0,
    # 【待校准参数】
    # 运行模式 3 后，根据建议修改这里的值
    "absolute_floor": 0.15,
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


# ===================== 2. 特征提取器 =====================
class ResNetFeatureExtractor(nn.Module):
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

    # === 核心逻辑提取：计算热力图 ===
    def get_heatmap(self, img_path):
        pre = robust_preprocess(img_path)
        if not pre:
            return None, None

        extractor.eval()
        with torch.no_grad():
            feat = extractor(pre["tensor"].to(DEVICE))
            feat = feat.cpu().numpy().astype(np.float32)[0]
            fh, fw = extractor.feature_map_h, extractor.feature_map_w

        # L2归一化
        norm = np.linalg.norm(feat, axis=1, keepdims=True) + 1e-8
        feat = feat / norm

        distances, _ = self.index.search(feat, CONFIG["n_neighbors"])
        score_map = np.mean(distances, axis=1).reshape(fh, fw)
        score_map = cv2.GaussianBlur(score_map, (3, 3), 0)

        h, w = CONFIG["input_size"]
        heatmap = cv2.resize(score_map, (w, h), interpolation=cv2.INTER_LINEAR)
        return heatmap, pre["processed_img"]

    # === 模式 3: 仅计算分数 ===
    def compute_score_only(self, img_path):
        heatmap, _ = self.get_heatmap(img_path)
        if heatmap is None:
            return 0.0
        return np.max(heatmap)

    # === 模式 2: 检测并保存 ===
    def detect_and_save(self, img_path, save_dir):
        heatmap, img = self.get_heatmap(img_path)
        if heatmap is None:
            return

        # 统计
        name = os.path.splitext(os.path.basename(img_path))[0]
        h_mean, h_std = np.mean(heatmap), np.std(heatmap)
        h_max = np.max(heatmap)

        # 阈值判定
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
                # 热力图
                hm_norm = (heatmap - heatmap.min()) / (
                    heatmap.max() - heatmap.min() + 1e-8
                )
                hm_color = cv2.applyColorMap(
                    (hm_norm * 255).astype(np.uint8), cv2.COLORMAP_JET
                )
                cv2.imwrite(os.path.join(save_dir, f"{name}_heat.jpg"), hm_color)
                return

        logging.info(f"✅ OK: {name}")


# ===================== 主程序 =====================
def main():
    print("🚀 智能布料检测系统 V5 (含校准模式)")
    DATASET_DIR = r"F:\\All_dataset\\\\long_buliao_datasets_260107_cutbackground"
    LIB_FILE = "PatchCoreDetection\\\\feature_lib\\\\lib_resnet18_260109_1.npz"
    TEST_DIR = r"F:\\All_dataset\\\\half_bad_half_good"
    # === 新增：校准用的正常图片目录 (通常就是训练集，或者另一批没瑕疵的图) ===
    CALIBRATION_DIR = r"F:\\All_dataset\\\\long_buliao_datasets_260107_cutbackground"

    print("\n请选择模式:")
    print("1: 构建新库 (Training)")
    print("2: 批量检测 (Testing)")
    print("3: 阈值校准 (Calibration - 统计正常图的最大分数)")

    action = input("选择: ")

    if action == "1":
        build_dynamic_library(DATASET_DIR, LIB_FILE)

    elif action == "2":
        if not os.path.exists(LIB_FILE):
            print("❌ 请先运行模式1")
            return
        detector = AutoDefectDetector(LIB_FILE)
        imgs = [
            os.path.join(r, f)
            for r, _, fs in os.walk(TEST_DIR)
            for f in fs
            if f.lower().endswith((".jpg", ".png"))
        ]
        print(f"📊 开始检测 {len(imgs)} 张图片...")
        for p in tqdm(imgs):
            detector.detect_and_save(p, "results_v5_final")

    elif action == "3":
        if not os.path.exists(LIB_FILE):
            print("❌ 请先运行模式1")
            return

        print(f"📂 正在加载库进行校准: {LIB_FILE}")
        detector = AutoDefectDetector(LIB_FILE)

        # 扫描校准文件夹
        calib_imgs = [
            os.path.join(r, f)
            for r, _, fs in os.walk(CALIBRATION_DIR)
            for f in fs
            if f.lower().endswith((".jpg", ".png"))
        ]

        # 随机抽取 50-100 张即可，不用全部跑完
        if len(calib_imgs) > 100:
            import random

            calib_imgs = random.sample(calib_imgs, 100)
            print(f"ℹ️ 图片太多，随机抽取 100 张进行统计...")

        print(f"📊 正在计算 {len(calib_imgs)} 张正常图片的分数分布...")
        max_scores = []

        for p in tqdm(calib_imgs):
            score = detector.compute_score_only(p)
            max_scores.append(score)

        # 统计结果
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

        # 给出建议
        suggested_floor = global_max + 0.05  # 留一点安全余量
        print(f"✅ 建议设置 absolute_floor = {suggested_floor:.2f}")
        print("=" * 40 + "\n")
        print(f"请修改 CONFIG 中的 'absolute_floor' 为上述建议值，然后运行模式 2。")


if __name__ == "__main__":
    main()
