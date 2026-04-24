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
from torchvision.models import convnext_small, ConvNeXt_Small_Weights
from torchvision.models.feature_extraction import create_feature_extractor
import logging

# ===================== 日志配置 =====================
file_handler = logging.FileHandler("defect_detection.log", mode="a", encoding="utf-8")
stream_handler = logging.StreamHandler()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[file_handler, stream_handler],
)

# ===================== 环境检查 =====================
try:
    import faiss

    FAISS_AVAILABLE = True
    logging.info("✅ FAISS已安装，将使用动态聚类索引 (IVF)")
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("⚠️ 未安装 FAISS，将回退到暴力 KNN（速度较慢且无法聚类）")

# ===================== 全局配置 =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===================== 全局配置 =====================
CONFIG = {
    "input_size": (384, 1024),
    "coreset_ratio": 0.05,
    "max_coreset_size": 40000,
    "batch_extract_size": 4,
    "n_neighbors": 9,  # 保持平滑
    "nprobe": 10,
    "clahe_clip_limit": 2.0,
    # === 关键修改：改用绝对分数阈值 ===
    # 既然之前归一化不好用，我们直接切原始距离。
    # 如何确定这个值？运行代码时，我会把每张图的最高分打印出来。
    # 你需要观察：好图的MaxScore是多少（比如5），坏图是多少（比如20）。
    # 然后取中间值（比如12）。
    "raw_threshold": 0.05,  # <--- 请根据日志输出来调整这个值！！！
    "min_defect_area": 80,  # 面积过滤
    "morph_kernel_size": 5,
}


# ===================== 1. 弹性尺寸预处理 =====================
def robust_preprocess(img_path):
    """
    处理高度波动的图像：统一缩放到 (1024, 384)
    """
    try:
        img_original = cv2.imread(img_path)
        if img_original is None:
            raise ValueError(f"无法读取: {img_path}")

        # 强制缩放到固定尺寸 (W=1024, H=384)
        # 对于 400 左右波动的图，这点形变是可以接受的，比 Padding 更好
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

        # 标准化
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

        return {
            "tensor": img_tensor,
            "processed_img": img_enhanced,
            "original_shape": img_original.shape[:2],
        }
    except Exception as e:
        logging.error(f"预处理错误: {e}")
        return None


# ===================== 2. 特征提取器 (不变) =====================
class MultiScaleFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        weights = ConvNeXt_Small_Weights.DEFAULT
        base_model = convnext_small(weights=weights)
        self.feature_extractor = create_feature_extractor(
            base_model, return_nodes={"features.3": "s2", "features.5": "s3"}
        )
        self.avg_pool = nn.AvgPool2d(3, stride=1, padding=1)
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


extractor = MultiScaleFeatureExtractor().to(DEVICE)


# ===================== 3. 智能构建特征库 (修复内存溢出版) =====================
def build_dynamic_library(normal_dir, save_path):
    logging.info("🚀 开始构建动态特征库 (内存优化版)...")

    img_paths = [
        os.path.join(r, f)
        for r, _, fs in os.walk(normal_dir)
        for f in fs
        if f.lower().endswith(("jpg", "png"))
    ]
    logging.info(f"📁 发现图像: {len(img_paths)} 张")

    if len(img_paths) == 0:
        logging.warning("⚠️ 未找到图像，无法构建库")
        return

    # --- 内存优化策略 ---
    # 我们不需要保存所有几千万个Patch特征。
    # 为了避免33GB内存溢出，我们设定一个中间缓冲区上限 (比如 100万个特征点，约占用 2GB 内存，非常安全)
    TARGET_INTERMEDIATE_SIZE = 1000000

    # 估算每张图的特征数 (根据报错信息约为 6144)
    # 动态计算采样率：如果图很多，每张图就少取一点；图少就多取一点
    est_total_patches = len(img_paths) * 6144
    sampling_ratio = TARGET_INTERMEDIATE_SIZE / est_total_patches
    sampling_ratio = min(1.0, max(0.02, sampling_ratio))  # 限制在 2% 到 100% 之间

    logging.info(f"📉 启用流式采样: 采样率约 {sampling_ratio:.2%} (避免内存爆炸)")

    # --- 1. 提取并即时采样 ---
    features_list = []
    extractor.eval()

    for i in tqdm(
        range(0, len(img_paths), CONFIG["batch_extract_size"]), desc="提取特征"
    ):
        batch = []
        for p in img_paths[i : i + CONFIG["batch_extract_size"]]:
            res = robust_preprocess(p)
            if res:
                batch.append(res["tensor"])

        if not batch:
            continue

        with torch.no_grad():
            feat = extractor(torch.cat(batch).to(DEVICE))  # 输出形状 [B, H*W, C]

        # === 关键修改：在循环内直接转 Numpy 并采样 ===
        feat_np = feat.cpu().numpy()

        # 展平: [B * (H*W), C]
        batch_features = feat_np.reshape(-1, feat_np.shape[-1])

        # 随机采样
        n_patches = batch_features.shape[0]
        n_sample = int(n_patches * sampling_ratio)
        n_sample = max(10, n_sample)  # 保证至少采几个点

        if n_sample < n_patches:
            indices = np.random.choice(n_patches, n_sample, replace=False)
            features_list.append(batch_features[indices])
        else:
            features_list.append(batch_features)

    if not features_list:
        logging.error("❌ 未提取到任何特征")
        return

    # 此时拼接的数据量已经大大减小，不会爆内存
    all_features = np.concatenate(features_list, axis=0)

    # 归一化
    all_features = all_features / (
        np.linalg.norm(all_features, axis=1, keepdims=True) + 1e-8
    )

    # --- 2. 二次精简 (KMeans Coreset) ---
    logging.info(
        f"📊 收集到中间特征: {all_features.shape[0]} (目标库大小: {CONFIG['max_coreset_size']})"
    )

    target_size = CONFIG["max_coreset_size"]

    # 如果中间特征还是比目标多太多，再用 KMeans 聚类精选
    if all_features.shape[0] > target_size:
        from sklearn.cluster import MiniBatchKMeans

        logging.info("⏳ 正在运行聚类压缩...")
        n_sampling_clusters = min(target_size // 5, 2000)
        kmeans = MiniBatchKMeans(
            n_clusters=n_sampling_clusters, batch_size=4096, n_init="auto"
        )
        labels = kmeans.fit_predict(all_features)

        indices = []
        for i in range(n_sampling_clusters):
            cluster_idx = np.where(labels == i)[0]
            if len(cluster_idx) > 0:
                sample_count = max(1, int(target_size / n_sampling_clusters))
                if len(cluster_idx) > sample_count:
                    indices.extend(
                        np.random.choice(cluster_idx, sample_count, replace=False)
                    )
                else:
                    indices.extend(cluster_idx)

        core_features = all_features[indices]
    else:
        core_features = all_features

    logging.info(f"✅ 最终特征库大小: {core_features.shape}")

    # --- 3. 动态构建 IVF 索引 ---
    if FAISS_AVAILABLE:
        d = core_features.shape[1]
        n_features = core_features.shape[0]
        nlist = int(4 * math.sqrt(n_features))
        nlist = max(16, min(nlist, 1024))

        logging.info(f"🧠 智能判断聚类数量: {nlist}")

        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
        index.train(core_features)
        index.add(core_features)

        index_path = save_path.replace(".npz", ".ivf_index")
        faiss.write_index(index, index_path)
        logging.info(f"✅ IVF索引已构建并保存")

    np.savez(save_path, features=core_features)


# ===================== 4. 检测器 (两阶段匹配) =====================
class AutoDefectDetector:
    def __init__(self, lib_path):
        self.lib_path = lib_path
        self.load_index()

    def load_index(self):
        idx_path = self.lib_path.replace(".npz", ".ivf_index")
        if FAISS_AVAILABLE and os.path.exists(idx_path):
            self.index = faiss.read_index(idx_path)
            self.index.nprobe = CONFIG["nprobe"]
            logging.info(f"✅ 加载检测器: 包含 {self.index.nlist} 种纹理原型")
            if DEVICE.type == "cuda" and hasattr(faiss, "StandardGpuResources"):
                try:
                    res = faiss.StandardGpuResources()
                    self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                except:
                    pass
        else:
            raise RuntimeError("未找到索引文件")

    def detect(self, img_path, save_dir):
        # 1. 预处理
        pre = robust_preprocess(img_path)
        if not pre:
            return

        # 2. 提取特征
        extractor.eval()
        with torch.no_grad():
            feat = extractor(pre["tensor"].to(DEVICE))
            feat = feat.cpu().numpy().astype(np.float32)[0]
            fh, fw = extractor.feature_map_h, extractor.feature_map_w

        # 归一化特征向量
        feat = feat / (np.linalg.norm(feat, axis=1, keepdims=True) + 1e-8)

        # 3. 搜索距离
        distances, _ = self.index.search(feat, CONFIG["n_neighbors"])

        # 4. 生成原始分数图 (不归一化!)
        score_map = np.mean(distances, axis=1).reshape(fh, fw)
        score_map = cv2.GaussianBlur(score_map, (3, 3), 0)  # 高斯平滑

        h, w = CONFIG["input_size"]
        heatmap = cv2.resize(score_map, (w, h), interpolation=cv2.INTER_LINEAR)

        # 5. 保存结果
        self.save_result(img_path, pre["processed_img"], heatmap, save_dir)

    def save_result(self, path, img, heatmap, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        name = os.path.splitext(os.path.basename(path))[0]

        # === 核心调试信息 ===
        # 打印这张图的“最大异常分数”。
        # 你需要盯着这个数字看：
        # 如果是好图，这个数应该很小；如果是坏图，这个数应该很大。
        max_score = np.max(heatmap)
        logging.info(f"🖼️ {name} | 最大异常分数: {max_score:.4f}")

        # === 绝对阈值分割 ===
        # 直接判断：超过 raw_threshold 才是瑕疵
        mask = (heatmap > CONFIG["raw_threshold"]).astype(np.uint8) * 255

        # 形态学去噪
        k_size = CONFIG["morph_kernel_size"]
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # 轮廓查找
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        res_img = img.copy()
        defect_count = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > CONFIG["min_defect_area"]:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(res_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                # 标出 面积 | 区域平均分数
                # 我们可以计算这个框内的平均异常分数，帮助你判断
                roi_mask = np.zeros_like(heatmap, dtype=np.uint8)
                cv2.drawContours(roi_mask, [cnt], -1, 1, -1)
                mean_val = cv2.mean(heatmap, mask=roi_mask)[0]

                label = f"A:{int(area)} S:{mean_val:.1f}"
                cv2.putText(
                    res_img,
                    label,
                    (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                )
                defect_count += 1

        if defect_count > 0:
            logging.info(f"❌ NG: {name} 发现 {defect_count} 处瑕疵")
            cv2.imwrite(os.path.join(save_dir, f"{name}_NG_res.jpg"), res_img)

            # 保存热力图 (为了可视化，这里还是需要归一化一下，但这不影响检测结果)
            heatmap_vis = (heatmap - heatmap.min()) / (
                heatmap.max() - heatmap.min() + 1e-8
            )
            hm_color = cv2.applyColorMap(
                (heatmap_vis * 255).astype(np.uint8), cv2.COLORMAP_JET
            )
            cv2.imwrite(os.path.join(save_dir, f"{name}_heat.jpg"), hm_color)
        else:
            logging.info(f"✅ OK: {name}")


# ===================== 主程序 =====================
def main():
    print("🚀 智能布料检测系统 (自适应尺寸 + 动态聚类)")

    # 请修改这里的路径
    DATASET_DIR = r"F:\All_dataset\\long_buliao_datasets_260107_cutbackground"
    LIB_FILE = "PatchCoreDetection\\feature_lib\\feature_lib_dynamic_260108_2.npz"
    TEST_DIR = r"F:\All_dataset\\half_bad_half_good"

    action = input("1: 构建库 (自动学习类别), 2: 批量检测\n选择: ")

    if action == "1":
        build_dynamic_library(DATASET_DIR, LIB_FILE)
    elif action == "2":
        if not os.path.exists(LIB_FILE):
            print(f"❌ 找不到特征库文件: {LIB_FILE}")
            print("请先运行模式 1 构建库")
            return

        print(f"📂 正在加载特征库: {LIB_FILE} ...")
        try:
            detector = AutoDefectDetector(LIB_FILE)
        except Exception as e:
            print(f"❌ 加载检测器失败: {e}")
            return

        print(f"🔍 正在扫描测试目录: {TEST_DIR}")

        # === 修改开始：支持更多格式且忽略大小写 ===
        valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")
        imgs = [
            os.path.join(r, f)
            for r, _, fs in os.walk(TEST_DIR)
            for f in fs
            if f.lower().endswith(valid_extensions)  # 使用 lower() 忽略大小写
        ]
        # === 修改结束 ===

        print(f"📊 共找到 {len(imgs)} 张测试图片")

        if len(imgs) == 0:
            print("⚠️ 警告：未找到任何图片！")
            print(f"1. 请确认路径是否存在: {os.path.exists(TEST_DIR)}")
            print(f"2. 请确认文件夹内是否有 {valid_extensions} 格式的图片")
            print("3. 请检查文件名是否为 .JPG (代码已自动处理大小写)")
        else:
            print("🚀 开始批量检测...")
            for p in tqdm(imgs):
                detector.detect(p, "results_dynamic")
            print(f"✅ 检测完成！结果保存在 results_dynamic 文件夹")


if __name__ == "__main__":
    main()
