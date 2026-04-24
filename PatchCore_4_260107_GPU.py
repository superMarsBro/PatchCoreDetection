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
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import convnext_small, ConvNeXt_Small_Weights
from torchvision.models.feature_extraction import create_feature_extractor
import logging

# ===================== 日志配置 =====================
file_handler = logging.FileHandler(
    "defect_detection_gpu.log", mode="a", encoding="utf-8"
)
stream_handler = logging.StreamHandler()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[file_handler, stream_handler],
)

# ===================== 环境检查 =====================
try:
    import faiss

    # 检查是否安装了 faiss-gpu
    if hasattr(faiss, "StandardGpuResources"):
        FAISS_GPU = True
        logging.info("✅ 检测到 faiss-gpu，将开启 GPU 极速聚类与搜索")
    else:
        FAISS_GPU = False
        logging.warning(
            "⚠️ 仅检测到 faiss-cpu，聚类速度受限。建议安装 pip install faiss-gpu"
        )
except ImportError:
    FAISS_GPU = False
    logging.warning("⚠️ 未安装 FAISS")

# ===================== 全局配置 =====================
# 开启 CuDNN 自动调优
torch.backends.cudnn.benchmark = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIG = {
    "input_size": (384, 1024),
    "coreset_ratio": 0.05,
    "max_coreset_size": 50000,  # GPU模式下可以适当增大库容量
    "batch_extract_size": 16,  # 开启 FP16 后，BatchSize 可以翻倍 (4 -> 8~16)
    "n_neighbors": 3,
    "nprobe": 10,
    "clahe_clip_limit": 2.0,
    "min_defect_area": 15,
}


# ===================== 1. 高效数据集加载 (CPU多进程预取) =====================
class ImageDataset(Dataset):
    def __init__(self, img_paths, input_size):
        self.img_paths = img_paths
        self.input_size = input_size  # (H, W)
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        try:
            # OpenCV 读取较快
            img = cv2.imread(path)
            if img is None:
                raise ValueError

            # Resize
            target_h, target_w = self.input_size
            img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

            # CLAHE
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            img = cv2.merge((l, a, b))
            img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)

            # 转 RGB 并 Transform
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            tensor = self.transform(img_rgb)

            return tensor, path
        except Exception as e:
            # 返回空 tensor 占位，后续过滤
            return torch.zeros(3, self.input_size[0], self.input_size[1]), "error"


# ===================== 2. 特征提取器 (FP16 加速) =====================
class MultiScaleFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        weights = ConvNeXt_Small_Weights.DEFAULT
        base_model = convnext_small(weights=weights)
        self.feature_extractor = create_feature_extractor(
            base_model, return_nodes={"features.3": "s2", "features.5": "s3"}
        )
        self.avg_pool = nn.AvgPool2d(3, stride=1, padding=1)

        self.feature_extractor.eval()
        self.feature_map_h = 0
        self.feature_map_w = 0

    def forward(self, x):
        # 确保输入不需要梯度
        with torch.no_grad():
            outputs = self.feature_extractor(x)
            s3_up = torch.nn.functional.interpolate(
                outputs["s3"], size=outputs["s2"].shape[-2:], mode="bilinear"
            )
            features = torch.cat([outputs["s2"], s3_up], dim=1)
            features = self.avg_pool(features)

            self.feature_map_h, self.feature_map_w = (
                features.shape[2],
                features.shape[3],
            )
            B, C, H, W = features.shape
            return features.permute(0, 2, 3, 1).reshape(B, H * W, C)


# 初始化模型并转为半精度 (FP16)
extractor = MultiScaleFeatureExtractor().to(DEVICE)
if torch.cuda.is_available():
    extractor.half()  # 开启 FP16 加速
    logging.info("⚡ 已开启 FP16 半精度加速")


# ===================== 3. GPU 加速构建特征库 =====================
def build_dynamic_library_gpu(normal_dir, save_path):
    logging.info("🚀 开始构建动态特征库 (GPU 加速版)...")

    img_paths = [
        os.path.join(r, f)
        for r, _, fs in os.walk(normal_dir)
        for f in fs
        if f.lower().endswith(("jpg", "png"))
    ]
    logging.info(f"📁 发现图像: {len(img_paths)} 张")

    if not img_paths:
        return

    # 使用 DataLoader 多进程读取
    dataset = ImageDataset(img_paths, CONFIG["input_size"])
    dataloader = DataLoader(
        dataset,
        batch_size=CONFIG["batch_extract_size"],
        shuffle=False,
        num_workers=4,  # CPU 核心数，负责预处理
        pin_memory=True,  # 加速 CPU -> GPU 传输
    )

    features_list = []

    # 流式采样参数
    TARGET_SIZE = 1000000
    est_patches = len(img_paths) * 6144
    sampling_ratio = min(1.0, max(0.01, TARGET_SIZE / est_patches))

    extractor.eval()

    for batch_tensors, paths in tqdm(dataloader, desc="GPU特征提取"):
        # 过滤错误图片
        valid_mask = [p != "error" for p in paths]
        if not any(valid_mask):
            continue
        batch_tensors = batch_tensors[valid_mask].to(DEVICE)

        # FP16 推理
        if torch.cuda.is_available():
            batch_tensors = batch_tensors.half()

        with torch.no_grad():
            feat = extractor(batch_tensors)

        feat_np = feat.float().cpu().numpy()  # Faiss 需要 float32

        # 批量采样
        B_size = feat_np.shape[0]
        for i in range(B_size):
            flat_f = feat_np[i]  # [H*W, C]
            n_p = flat_f.shape[0]
            n_s = int(n_p * sampling_ratio)
            if n_s > 0:
                idx = np.random.choice(n_p, n_s, replace=False)
                features_list.append(flat_f[idx])

    if not features_list:
        return

    all_features = np.concatenate(features_list, axis=0)
    # 归一化
    faiss.normalize_L2(all_features)

    logging.info(f"📊 收集特征总数: {all_features.shape[0]}")

    # --- GPU KMeans 聚类 (显存允许的情况下) ---
    target_size = CONFIG["max_coreset_size"]

    if all_features.shape[0] > target_size:
        logging.info("⚡ 使用 GPU KMeans 进行压缩...")
        n_clusters = min(target_size // 5, 2048)

        if FAISS_GPU:
            # 使用 Faiss 的 GPU KMeans，速度极快
            try:
                # 显存管理：如果数据太大，还是得切分或者用CPU
                res = faiss.StandardGpuResources()
                kmeans = faiss.Clustering(all_features.shape[1], n_clusters)
                kmeans.niter = 20
                kmeans.verbose = True

                # 将数据转到 GPU
                index_flat = faiss.IndexFlatL2(all_features.shape[1])
                gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)

                kmeans.train(all_features, gpu_index_flat)

                # 寻找最近的样本
                _, min_indices = gpu_index_flat.search(
                    kmeans.centroids, 1
                )  # 这里只是近似
                # 为了更准确的采样，还是建议用 CPU 的 MiniBatchKMeans 或者 Faiss 的采样逻辑
                # 这里为了稳健，如果显存足够，可以用 GPU index search

                # 简化策略：直接把聚类中心当作 Coreset (虽然不是最完美的 PatchCore 实现，但速度最快且效果接近)
                # 或者，为了标准 PatchCore，我们仍然回退到 CPU 采样，但特征已经少了很多
                pass
            except Exception as e:
                logging.warning(f"GPU KMeans 失败 ({e})，回退到 CPU")

        # 仍然推荐用 sklearn 的 MiniBatchKMeans，因为这一步不需要太频繁
        from sklearn.cluster import MiniBatchKMeans

        kmeans_sk = MiniBatchKMeans(
            n_clusters=n_clusters, batch_size=4096, n_init="auto"
        )
        labels = kmeans_sk.fit_predict(all_features)

        indices = []
        for i in range(n_clusters):
            c_idx = np.where(labels == i)[0]
            if len(c_idx) > 0:
                cnt = max(1, int(target_size / n_clusters))
                indices.extend(
                    np.random.choice(c_idx, min(len(c_idx), cnt), replace=False)
                )
        core_features = all_features[indices]
    else:
        core_features = all_features

    logging.info(f"✅ 最终库大小: {core_features.shape}")

    # --- GPU IVF 索引构建 ---
    if FAISS_GPU:
        logging.info("⚡ 构建 GPU IVF 索引...")
        d = core_features.shape[1]
        nlist = min(1024, int(4 * math.sqrt(core_features.shape[0])))

        res = faiss.StandardGpuResources()
        # 构建 GPU 索引配置
        config = faiss.GpuIndexIVFFlatConfig()
        config.useFloat16 = True  # 开启 FP16 索引，进一步省显存

        idx_gpu = faiss.GpuIndexIVFFlat(res, d, nlist, faiss.METRIC_L2, config)
        idx_gpu.train(core_features)
        idx_gpu.add(core_features)

        # 转回 CPU 保存
        index_cpu = faiss.index_gpu_to_cpu(idx_gpu)
        faiss.write_index(index_cpu, save_path.replace(".npz", ".ivf_index"))
    else:
        # CPU 降级
        pass

    np.savez(save_path, features=core_features)


# ===================== 4. GPU 加速检测器 =====================
class AutoDefectDetectorGPU:
    def __init__(self, lib_path):
        self.lib_path = lib_path
        self.load_index()

    def load_index(self):
        idx_path = self.lib_path.replace(".npz", ".ivf_index")
        if os.path.exists(idx_path):
            cpu_index = faiss.read_index(idx_path)
            cpu_index.nprobe = CONFIG["nprobe"]

            if FAISS_GPU:
                logging.info("⚡ 将索引加载到显存...")
                res = faiss.StandardGpuResources()
                # 转移到 GPU
                self.index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
            else:
                self.index = cpu_index
        else:
            raise FileNotFoundError("索引未找到")

    def detect(self, img_path, save_dir):
        # 1. 单张预处理 (OpenCV CPU) - 如果是批量检测建议也用 DataLoader
        # 这里为了兼容单张调用保持原样，但在 batch_detect 中我们会优化
        pass  # 具体逻辑在 batch_detect 中实现效率更高


# ===================== 5. 批量检测流水线 =====================
def run_batch_detection(detector, test_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    img_paths = [
        os.path.join(r, f)
        for r, _, fs in os.walk(test_dir)
        for f in fs
        if f.endswith(".jpg")
    ]

    dataset = ImageDataset(img_paths, CONFIG["input_size"])
    dataloader = DataLoader(
        dataset,
        batch_size=CONFIG["batch_extract_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    extractor.eval()

    total_defects = 0

    for batch_tensors, batch_paths in tqdm(dataloader, desc="GPU批量检测"):
        valid_idx = [i for i, p in enumerate(batch_paths) if p != "error"]
        if not valid_idx:
            continue

        tensors = batch_tensors[valid_idx].to(DEVICE)
        if torch.cuda.is_available():
            tensors = tensors.half()
        paths = [batch_paths[i] for i in valid_idx]

        # 1. 特征提取
        with torch.no_grad():
            feat = extractor(tensors)  # [B, H*W, C]
            fh, fw = extractor.feature_map_h, extractor.feature_map_w

        # 2. 向量搜索
        B = feat.shape[0]
        feat_flat = feat.view(B * fh * fw, -1).float()  # Faiss 需要 float32

        # L2 归一化 (在 GPU 上做)
        # 手动归一化比 faiss.normalize_L2 更灵活，避免修改原数据
        norm = torch.norm(feat_flat, p=2, dim=1, keepdim=True) + 1e-8
        feat_flat = feat_flat / norm

        # 搜索 (GPU)
        # inputs: tensor(GPU) -> numpy(CPU) -> faiss(GPU)
        # faiss-gpu 支持直接接收 torch gpu tensor (通过指针)，但 Python 接口通常需要 numpy
        # 为稳定起见，转 numpy
        feat_np = feat_flat.cpu().numpy()

        dists, _ = detector.index.search(feat_np, CONFIG["n_neighbors"])

        # 3. 后处理 (每个样本单独处理)
        dists = dists.mean(axis=1).reshape(B, fh, fw)

        for i in range(B):
            score_map = dists[i]
            img_path = paths[i]

            # 高斯模糊 (CPU OpenCV)
            score_map = cv2.GaussianBlur(score_map, (3, 3), 0)

            # 上采样
            h, w = CONFIG["input_size"]
            heatmap = cv2.resize(score_map, (w, h), interpolation=cv2.INTER_BILINEAR)

            # 简单可视化保存逻辑
            heatmap_norm = (heatmap - heatmap.min()) / (
                heatmap.max() - heatmap.min() + 1e-8
            )
            if heatmap_norm.max() > 0:  # 避免空
                thresh = np.percentile(heatmap_norm, 98)
                mask = (heatmap_norm > thresh).astype(np.uint8) * 255
                contours, _ = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                has_defect = False
                img_viz = cv2.imread(img_path)
                img_viz = cv2.resize(img_viz, (w, h))  # 保持和heatmap一致

                for cnt in contours:
                    if cv2.contourArea(cnt) > CONFIG["min_defect_area"]:
                        x, y, bw, bh = cv2.boundingRect(cnt)
                        cv2.rectangle(img_viz, (x, y), (x + bw, y + bh), (0, 0, 255), 2)
                        has_defect = True

                if has_defect:
                    total_defects += 1
                    name = os.path.basename(img_path)
                    cv2.imwrite(os.path.join(save_dir, f"res_{name}"), img_viz)

    logging.info(f"检测完成，共发现 {total_defects} 张瑕疵图")


# ===================== 主程序 =====================
def main():
    print("🚀 GPU 极速布料检测系统 (FP16 + Faiss-GPU + DataLoader)")
    DATASET_DIR = r"F:\All_dataset\\long_buliao_datasets_260107_cutbackground"
    LIB_FILE = "PatchCoreDetection\\feature_lib\\feature_lib_gpu_260107.npz"
    TEST_DIR = r"F:\All_dataset\\half_bad_half_good"

    action = input("1: 构建库, 2: 批量检测\n选择: ")

    if action == "1":
        build_dynamic_library_gpu(DATASET_DIR, LIB_FILE)
    elif action == "2":
        detector = AutoDefectDetectorGPU(LIB_FILE)
        run_batch_detection(detector, TEST_DIR, "results_gpu")


if __name__ == "__main__":
    main()
