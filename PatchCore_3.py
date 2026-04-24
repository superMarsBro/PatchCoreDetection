import torch
import torch.nn as nn
import cv2
import numpy as np
import os
import warnings
import json
import time
from PIL import Image
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from torchvision import transforms
from torchvision.models import convnext_small, ConvNeXt_Small_Weights
from scipy import ndimage
from multiprocessing import Pool, cpu_count
import logging

# 配置日志
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s",
#     handlers=[logging.FileHandler("defect_detection.log"), logging.StreamHandler()],
# )

# 修改点：在 FileHandler 中显式指定 encoding='utf-8'
file_handler = logging.FileHandler("defect_detection.log", mode="a", encoding="utf-8")
stream_handler = logging.StreamHandler()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[file_handler, stream_handler],
)


# 尝试导入FAISS（大样本KNN加速）
try:
    import faiss

    FAISS_AVAILABLE = True
    logging.info("✅ FAISS已安装，将使用高效索引")
except ImportError:
    FAISS_AVAILABLE = False
    warnings.warn("未安装FAISS，将使用scikit-learn KNN（大样本下速度较慢）")
    logging.warning("⚠️ FAISS未安装，建议安装以提升大样本处理速度")

# ===================== 全局配置优化（适配6000+样本） =====================
IMG_SIZE = (512, 512)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"💻 使用设备: {DEVICE}")

# 核心配置（针对6000+正常样本和像素级检测优化）
CONFIG = {
    # 特征库构建（大样本适配）
    "n_neighbors": 15,  # 邻居数（平衡精度和速度）
    "min_normal_samples": 100,  # 最小正常样本数（6000+足够）
    "coreset_ratio": 0.02,  # Coreset采样比例（6000×0.02=1200，平衡内存和精度）
    "max_coreset_size": 3000,  # 最大核心特征数（大样本下适当增加）
    "batch_extract_size": 32,  # 批量特征提取大小（GPU显存适配）
    "feature_normalize": True,  # 特征归一化（提升匹配精度）
    # 预处理（像素级保真）
    "clahe_clip_limit": 1.2,  # 温和的CLAHE（保留像素细节）
    "denoise_strength": 5,  # 轻量去噪（避免丢失微小瑕疵）
    # 像素级检测参数
    "threshold_std_multiplier": 3.0,  # 异常阈值（像素级检测更敏感）
    "min_defect_area": 10,  # 最小缺陷面积（像素级，更小阈值）
    "texture_variance_threshold": 20,  # 纹理方差阈值（像素级纹理分析）
    "use_adaptive_threshold": True,  # 自适应阈值（适配不同区域）
    "heatmap_upscale_mode": "bilinear",  # 热力图上采样模式（像素级对齐）
    # 后处理（像素级优化）
    "morphology_kernel_size": 3,  # 小核形态学操作（保留像素级细节）
    "contour_approximation_eps": 0.01,  # 轮廓逼近精度（像素级）
    "save_pixel_mask": True,  # 保存像素级瑕疵掩码
    "output_pixel_coords": True,  # 输出瑕疵像素坐标
}

logging.info(f"🔧 最终配置: {json.dumps(CONFIG, indent=2, ensure_ascii=False)}")


# ===================== 像素级预处理（保留细节） =====================
def pixel_aware_preprocess(img_path):
    """
    像素级感知的预处理：保留微小瑕疵细节，同时归一化
    返回：处理后的图像（用于可视化）、张量（用于特征提取）、原始尺寸（用于像素映射）
    """
    try:
        # 读取图像（保留原始尺寸信息）
        img_original = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img_original is None:
            raise ValueError(f"图像读取失败: {img_path}")
        orig_h, orig_w = img_original.shape[:2]
        h, w = img_original.shape[:2]
        scale = min(IMG_SIZE[0] / h, IMG_SIZE[1] / w)
        new_w, new_h = int(w * scale), int(h * scale)
        # 保持比例缩放
        resized_content = cv2.resize(
            img_original, (new_w, new_h), interpolation=cv2.INTER_LINEAR
        )
        # 创建黑色背景 (Canvas)
        img_resized = np.zeros((IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.uint8)
        # 居中粘贴
        y_off = (IMG_SIZE[0] - new_h) // 2
        x_off = (IMG_SIZE[1] - new_w) // 2
        img_resized[y_off : y_off + new_h, x_off : x_off + new_w] = resized_content

        # 1. 尺寸归一化（保持像素比例）
        # img_resized = cv2.resize(img_original, IMG_SIZE, interpolation=cv2.INTER_LINEAR)
        # 改进为保持比例缩放

        # 2. 像素级光照归一化（LAB空间，保留细节）
        lab = cv2.cvtColor(img_resized, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(
            clipLimit=CONFIG["clahe_clip_limit"], tileGridSize=(8, 8)
        )
        l_enhanced = clahe.apply(l)
        lab_enhanced = cv2.merge((l_enhanced, a, b))
        img_normalized = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

        # 3. 轻量级去噪（保留微小瑕疵）
        img_denoised = cv2.fastNlMeansDenoisingColored(
            img_normalized,
            None,
            h=CONFIG["denoise_strength"],
            hColor=CONFIG["denoise_strength"],
            templateWindowSize=7,
            searchWindowSize=15,
        )

        # 4. 张量转换（标准化）
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        img_rgb = cv2.cvtColor(img_denoised, cv2.COLOR_BGR2RGB)
        img_tensor = transform(img_rgb).unsqueeze(0)

        return {
            "processed_img": img_denoised,
            "tensor": img_tensor,
            "original_size": (orig_h, orig_w),
            "resized_size": IMG_SIZE,
            "path": img_path,
        }
    except Exception as e:
        logging.error(f"预处理失败 {img_path}: {str(e)}")
        raise


# ===================== 特征提取器（优化大样本提取） =====================
class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        weights = ConvNeXt_Small_Weights.DEFAULT
        self.backbone = convnext_small(weights=weights)
        # 移除分类头，保留特征提取部分（输出16×16×768特征图）
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        # 冻结参数
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

    def forward(self, x):
        features = self.backbone(x)
        # 特征维度：B×768×16×16 → 重塑为B×(16×16)×768
        B, C, H, W = features.shape
        features = features.permute(0, 2, 3, 1).reshape(B, H * W, C)
        return features


# 初始化特征提取器
extractor = FeatureExtractor().to(DEVICE)
logging.info("✅ 特征提取器初始化完成（ConvNeXt-Small）")


# ===================== 大样本特征提取（批量处理） =====================
def batch_extract_features(img_paths):
    """
    批量提取特征（适配6000+样本）
    """
    features_list = []
    extractor.eval()

    # 分批次处理
    for i in tqdm(
        range(0, len(img_paths), CONFIG["batch_extract_size"]),
        desc="批量提取特征",
        leave=False,
    ):
        batch_paths = img_paths[i : i + CONFIG["batch_extract_size"]]
        batch_tensors = []

        # 预处理批次图像
        for path in batch_paths:
            try:
                preprocess_result = pixel_aware_preprocess(path)
                batch_tensors.append(preprocess_result["tensor"])
            except Exception as e:
                logging.warning(f"跳过图像 {path}: {str(e)}")
                continue

        if not batch_tensors:
            continue

        # 批量前向传播
        batch_tensor = torch.cat(batch_tensors, dim=0).to(DEVICE)
        with torch.no_grad():
            batch_features = extractor(batch_tensor)

        # 收集特征
        batch_features_np = batch_features.cpu().numpy().astype(np.float32)
        for feat in batch_features_np:
            features_list.append(feat)

    if not features_list:
        raise ValueError("没有提取到有效特征")

    # 合并所有特征
    all_features = np.concatenate(features_list, axis=0)

    # 特征归一化（提升匹配精度）
    if CONFIG["feature_normalize"]:
        all_features = all_features / (
            np.linalg.norm(all_features, axis=1, keepdims=True) + 1e-8
        )

    logging.info(f"✅ 特征提取完成，总特征数: {all_features.shape[0]}")
    return all_features


# ===================== 优化的Coreset采样（大样本适配） =====================
def large_scale_coreset_sampling(features):
    """
    大样本下的Coreset采样：平衡速度和代表性
    """
    n_total = features.shape[0]
    target_size = min(
        CONFIG["max_coreset_size"], int(n_total * CONFIG["coreset_ratio"])
    )

    if target_size >= n_total:
        logging.info("📊 无需采样，使用全部特征")
        return features

    # 1. 先进行特征层面的 L2 归一化 (PatchCore标准步骤)
    features = features / np.linalg.norm(features, axis=1, keepdims=True)

    # 大样本下使用快速随机采样+聚类优化
    logging.info(f"🎯 大样本Coreset采样: {n_total} → {target_size}")

    # 第一步：快速随机采样（减少计算量）
    # if n_total > 10000:
    #     sample_size = min(5000, n_total)
    #     random_idx = np.random.choice(n_total, sample_size, replace=False)
    #     sample_features = features[random_idx]
    # else:
    #     sample_features = features

    # 2. 改进的采样：不要扔掉太多
    if n_total > 20000:
        # 保留 10% 或者至少 20000 个，而不是只留 5000
        sample_ratio = 0.1
        pre_sample_size = max(20000, int(n_total * sample_ratio))
        random_idx = np.random.choice(n_total, pre_sample_size, replace=False)
        sample_features = features[random_idx]
    else:
        sample_features = features

    logging.info(f"聚类前特征数: {sample_features.shape[0]}")

    # 第二步：KMeans聚类采样（保证代表性）
    from sklearn.cluster import MiniBatchKMeans

    n_clusters = min(target_size // 2, 500)
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters, batch_size=1000, n_init=3, random_state=42
    )
    cluster_labels = kmeans.fit_predict(sample_features)

    # 从每个聚类中采样
    sampled_indices = []
    samples_per_cluster = max(1, target_size // n_clusters)

    for cluster_id in range(n_clusters):
        cluster_mask = cluster_labels == cluster_id
        cluster_feats_idx = np.where(cluster_mask)[0]

        if len(cluster_feats_idx) == 0:
            continue

        # 采样
        if len(cluster_feats_idx) <= samples_per_cluster:
            sampled_indices.extend(cluster_feats_idx)
        else:
            selected = np.random.choice(
                cluster_feats_idx, samples_per_cluster, replace=False
            )
            sampled_indices.extend(selected)

    # 最终采样特征
    sampled_features = sample_features[sampled_indices]
    logging.info(f"✅ Coreset采样完成，最终特征数: {sampled_features.shape[0]}")
    return sampled_features


# ===================== 大样本特征库构建 =====================
def build_large_scale_feature_library(
    normal_dir, save_path="large_scale_feature_lib.npz"
):
    """
    构建适配6000+样本的特征库
    """
    logging.info("🚀 开始构建大样本特征库")

    # 1. 收集所有正常样本路径
    normal_paths = []
    for root, _, files in os.walk(normal_dir):
        for file in files:
            if file.lower().endswith(("jpg", "jpeg", "png", "bmp")):
                normal_paths.append(os.path.join(root, file))

    n_samples = len(normal_paths)
    if n_samples < CONFIG["min_normal_samples"]:
        raise ValueError(
            f"正常样本数量不足: {n_samples} < {CONFIG['min_normal_samples']}"
        )
    logging.info(f"📁 找到 {n_samples} 张正常样本图像")

    # 2. 批量提取特征
    all_features = batch_extract_features(normal_paths)

    # 3. 异常特征过滤（基于范数）
    logging.info("🔍 过滤异常特征")
    feature_norms = np.linalg.norm(all_features, axis=1)
    norm_mean = np.mean(feature_norms)
    norm_std = np.std(feature_norms)

    # 过滤极端范数特征
    lower_bound = norm_mean - 2.0 * norm_std
    upper_bound = norm_mean + 2.0 * norm_std
    valid_mask = (feature_norms >= lower_bound) & (feature_norms <= upper_bound)
    filtered_features = all_features[valid_mask]
    logging.info(
        f"✅ 特征过滤完成: {all_features.shape[0]} → {filtered_features.shape[0]}"
    )

    # 4. Coreset采样
    core_features = large_scale_coreset_sampling(filtered_features)

    # 5. 构建高效索引（FAISS优先）
    if FAISS_AVAILABLE:
        # 使用FAISS构建索引
        index = faiss.IndexFlatL2(core_features.shape[1])
        if DEVICE.type == "cuda":
            # GPU加速
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        index.add(core_features)
        # 保存索引
        faiss.write_index(
            faiss.index_gpu_to_cpu(index) if DEVICE.type == "cuda" else index,
            save_path.replace(".npz", "_faiss.index"),
        )
        logging.info("✅ FAISS索引已保存")
    else:
        # 使用sklearn KNN
        knn = NearestNeighbors(
            n_neighbors=CONFIG["n_neighbors"],
            metric="euclidean",
            n_jobs=min(8, cpu_count()),
        )
        knn.fit(core_features)

    # 6. 保存特征库
    save_data = {
        "features": core_features,
        "config": CONFIG,
        "n_total_samples": n_samples,
        "extract_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    np.savez(save_path, **save_data)
    logging.info(f"💾 特征库已保存至: {save_path}")

    return core_features


# ===================== 加载特征库 =====================
def load_large_scale_feature_library(load_path="large_scale_feature_lib.npz"):
    """
    加载大样本特征库
    """
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"特征库文件不存在: {load_path}")

    # 加载特征数据
    data = np.load(load_path, allow_pickle=True)
    core_features = data["features"]
    logging.info(f"✅ 加载特征库，特征数: {core_features.shape[0]}")

    # 加载索引
    index_path = load_path.replace(".npz", "_faiss.index")
    if FAISS_AVAILABLE and os.path.exists(index_path):
        # 加载FAISS索引
        index = faiss.read_index(index_path)
        if DEVICE.type == "cuda":
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        logging.info("✅ 加载FAISS索引（GPU加速）")
        return index, core_features
    else:
        # 使用sklearn KNN
        knn = NearestNeighbors(
            n_neighbors=CONFIG["n_neighbors"],
            metric="euclidean",
            n_jobs=min(8, cpu_count()),
        )
        knn.fit(core_features)
        logging.info("✅ 加载sklearn KNN索引")
        return knn, core_features


# ===================== 像素级瑕疵检测器 =====================
class PixelLevelDefectDetector:
    """
    像素级布料瑕疵检测器
    """

    def __init__(self, index, feature_lib):
        self.index = index
        self.feature_lib = feature_lib
        self.use_faiss = FAISS_AVAILABLE
        self.patch_size = (16, 16)  # 特征图patch大小（对应原图32×32像素）
        self.feature_dim = feature_lib.shape[1]

    def compute_pixel_anomaly_scores(self, img_tensor):
        """
        计算像素级异常分数
        """
        # 提取特征
        extractor.eval()
        with torch.no_grad():
            img_tensor = img_tensor.to(DEVICE)
            features = extractor(img_tensor)  # B×256×768
            features_np = features.cpu().numpy().astype(np.float32)[0]  # 256×768

        # 特征归一化（与训练时保持一致）
        if CONFIG["feature_normalize"]:
            features_np = features_np / (
                np.linalg.norm(features_np, axis=1, keepdims=True) + 1e-8
            )

        # 计算最近邻距离
        if self.use_faiss:
            # FAISS查询
            distances, _ = self.index.search(features_np, CONFIG["n_neighbors"])
        else:
            # sklearn KNN查询
            distances, _ = self.index.kneighbors(features_np)

        # 鲁棒的异常分数计算（中位数更稳定）
        median_distances = np.median(distances, axis=1)
        # 分数归一化到[0,1]
        max_dist = np.max(median_distances)
        if max_dist > 0:
            median_distances = median_distances / max_dist

        return median_distances

    def generate_pixel_heatmap(self, patch_scores):
        """
        生成像素级热力图（精确对齐原图）
        """
        # 重塑为patch网格
        patch_h, patch_w = 16, 16
        score_map = patch_scores.reshape(patch_h, patch_w)

        # 多尺度高斯滤波（保留像素级细节）
        score_map = cv2.GaussianBlur(score_map, (3, 3), sigmaX=0.8, sigmaY=0.8)

        # 像素级上采样（双线性插值，保持精度）
        upscale_methods = {
            "nearest": cv2.INTER_NEAREST,
            "bilinear": cv2.INTER_LINEAR,
            "bicubic": cv2.INTER_CUBIC,
        }
        heatmap = cv2.resize(
            score_map,
            IMG_SIZE,
            interpolation=upscale_methods[CONFIG["heatmap_upscale_mode"]],
        )

        return heatmap

    def pixel_level_post_process(self, heatmap, processed_img):
        """
        像素级后处理：精准提取瑕疵区域
        """
        # 1. 自适应阈值（像素级）
        if CONFIG["use_adaptive_threshold"]:
            # 全局阈值 + 局部自适应阈值
            global_mean = np.mean(heatmap)
            global_std = np.std(heatmap)
            global_thresh = (
                global_mean + CONFIG["threshold_std_multiplier"] * global_std
            )

            # 局部自适应阈值（16×16像素块）
            local_thresh = (
                cv2.adaptiveThreshold(
                    (heatmap * 255).astype(np.uint8),
                    255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY,
                    blockSize=33,
                    C=2,
                )
                / 255.0
            )
            local_thresh_mean = (
                np.mean(local_thresh[local_thresh > 0])
                if np.any(local_thresh > 0)
                else global_thresh
            )

            # 综合阈值
            final_thresh = min(global_thresh, local_thresh_mean)
        else:
            final_thresh = np.percentile(heatmap, 95)

        # 2. 二值化（像素级）
        binary_mask = (heatmap > final_thresh).astype(np.uint8) * 255

        # 3. 像素级形态学操作（小核）
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (CONFIG["morphology_kernel_size"], CONFIG["morphology_kernel_size"]),
        )
        # 闭运算（填充小空洞）
        binary_mask = cv2.morphologyEx(
            binary_mask, cv2.MORPH_CLOSE, kernel, iterations=1
        )
        # 开运算（去除小噪点）
        binary_mask = cv2.morphologyEx(
            binary_mask, cv2.MORPH_OPEN, kernel, iterations=1
        )

        # 4. 像素级连通域分析
        contours, hierarchy = cv2.findContours(
            binary_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE,  # 保留所有轮廓点（像素级）
        )

        # 5. 过滤小区域和无效轮廓
        valid_contours = []
        defect_pixel_info = []

        for cnt_idx, cnt in enumerate(contours):
            # 像素级面积计算
            area = cv2.contourArea(cnt)
            if area < CONFIG["min_defect_area"]:
                continue

            # 轮廓逼近（像素级精度）
            epsilon = CONFIG["contour_approximation_eps"] * cv2.arcLength(cnt, True)
            approx_cnt = cv2.approxPolyDP(cnt, epsilon, True)

            # 计算边界框（像素级坐标）
            x, y, w, h = cv2.boundingRect(cnt)
            # 计算最小包围圆（像素级）
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)

            # 提取瑕疵区域的所有像素坐标
            if CONFIG["output_pixel_coords"]:
                # 创建掩码并提取像素坐标
                cnt_mask = np.zeros(binary_mask.shape, dtype=np.uint8)
                cv2.drawContours(cnt_mask, [cnt], 0, 255, -1)
                defect_pixels = np.argwhere(cnt_mask == 255)
                # 转换为(x,y)格式
                defect_pixels = [(int(p[1]), int(p[0])) for p in defect_pixels]
            else:
                defect_pixels = []

            # 瑕疵类型判断（基于像素级特征）
            aspect_ratio = w / h if h > 0 else 1.0
            circularity = 4 * np.pi * area / (cv2.arcLength(cnt, True) ** 2 + 1e-8)

            if aspect_ratio > 5 or aspect_ratio < 0.2:
                defect_type = "划痕"
            elif circularity > 0.7:
                defect_type = "污点"
            elif area > 200:
                defect_type = "大瑕疵"
            else:
                defect_type = "小瑕疵"

            # 计算该区域的平均异常分数
            roi_heatmap = heatmap[y : y + h, x : x + w]
            avg_score = np.mean(roi_heatmap) if roi_heatmap.size > 0 else 0

            valid_contours.append(approx_cnt)
            defect_pixel_info.append(
                {
                    "defect_id": cnt_idx + 1,
                    "type": defect_type,
                    "pixel_area": int(area),
                    "bbox_pixels": (int(x), int(y), int(x + w), int(y + h)),
                    "center_pixel": (int(cx), int(cy)),
                    "radius_pixel": int(radius),
                    "avg_anomaly_score": float(avg_score),
                    "pixel_coords": defect_pixels[:1000],  # 限制保存的像素数量
                    "total_pixels": len(defect_pixels),
                }
            )

        # 6. 生成像素级结果图
        result_img = processed_img.copy()
        # 绘制像素级轮廓和信息
        color_map = {
            "划痕": (0, 255, 255),
            "污点": (0, 0, 255),
            "大瑕疵": (255, 0, 0),
            "小瑕疵": (0, 255, 0),
        }

        for idx, (cnt, info) in enumerate(zip(valid_contours, defect_pixel_info)):
            color = color_map[info["type"]]
            # 绘制轮廓（像素级）
            cv2.drawContours(result_img, [cnt], -1, color, 1)
            # 绘制边界框
            x1, y1, x2, y2 = info["bbox_pixels"]
            cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
            # 绘制中心点
            cx, cy = info["center_pixel"]
            cv2.circle(result_img, (cx, cy), 3, (255, 255, 255), -1)
            # 添加文本信息
            text = f"{info['type']}({info['pixel_area']}px)"
            cv2.putText(
                result_img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
            )

        # 生成干净的像素级瑕疵掩码
        clean_mask = np.zeros_like(binary_mask)
        cv2.drawContours(clean_mask, valid_contours, -1, 255, -1)

        return {
            "result_img": result_img,
            "defect_mask": clean_mask,
            "heatmap": heatmap,
            "defect_info": defect_pixel_info,
            "threshold": final_thresh,
            "n_defects": len(valid_contours),
        }

    def detect_pixel_level_defects(self, img_path, save_dir="pixel_level_results"):
        """
        像素级瑕疵检测主函数
        """
        logging.info(f"🔍 开始像素级检测: {os.path.basename(img_path)}")
        start_time = time.time()

        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        try:
            # 1. 像素级预处理
            preprocess_result = pixel_aware_preprocess(img_path)
            processed_img = preprocess_result["processed_img"]
            img_tensor = preprocess_result["tensor"]

            # 2. 计算像素级异常分数
            patch_scores = self.compute_pixel_anomaly_scores(img_tensor)

            # 3. 生成像素级热力图
            heatmap = self.generate_pixel_heatmap(patch_scores)

            # 4. 像素级后处理
            post_result = self.pixel_level_post_process(heatmap, processed_img)

            # 5. 保存结果（像素级）
            # 保存检测结果图
            cv2.imwrite(
                os.path.join(save_dir, f"{img_name}_result.jpg"),
                post_result["result_img"],
            )
            # 保存像素级瑕疵掩码
            if CONFIG["save_pixel_mask"]:
                cv2.imwrite(
                    os.path.join(save_dir, f"{img_name}_defect_mask.png"),
                    post_result["defect_mask"],
                )
            # 保存热力图（彩色）
            heatmap_normalized = (heatmap - heatmap.min()) / (
                heatmap.max() - heatmap.min() + 1e-8
            )
            heatmap_colored = cv2.applyColorMap(
                (heatmap_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET
            )
            cv2.imwrite(
                os.path.join(save_dir, f"{img_name}_heatmap.jpg"), heatmap_colored
            )

            # 6. 保存像素级检测报告（JSON）
            detection_report = {
                "image_path": img_path,
                "original_size": preprocess_result["original_size"],
                "processed_size": preprocess_result["resized_size"],
                "detection_time_seconds": round(time.time() - start_time, 3),
                "threshold": float(post_result["threshold"]),
                "n_defects": post_result["n_defects"],
                "defect_details": post_result["defect_info"],
                "config": CONFIG,
            }

            with open(
                os.path.join(save_dir, f"{img_name}_detection_report.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(detection_report, f, ensure_ascii=False, indent=2)

            # 打印结果摘要
            logging.info(f"✅ 检测完成: {img_name}")
            logging.info(f"   耗时: {time.time() - start_time:.3f}秒")
            logging.info(f"   检测到瑕疵数: {post_result['n_defects']}")
            logging.info(f"   检测阈值: {post_result['threshold']:.4f}")

            for defect in post_result["defect_info"]:
                logging.info(
                    f"   瑕疵{defect['defect_id']}: {defect['type']}, 面积{defect['pixel_area']}像素, "
                    f"位置{defect['center_pixel']}"
                )

            return detection_report

        except Exception as e:
            logging.error(f"检测失败 {img_path}: {str(e)}")
            return None


# ===================== 批量检测工具 =====================
def batch_detect_pixel_level(
    defector, test_dir, save_dir="batch_pixel_results260107-2", max_images=None
):
    """
    批量像素级检测
    """
    logging.info(f"📁 开始批量像素级检测: {test_dir}")
    os.makedirs(save_dir, exist_ok=True)

    # 收集图像路径
    img_paths = []
    for root, _, files in os.walk(test_dir):
        for file in files:
            if file.lower().endswith(("jpg", "jpeg", "png", "bmp")):
                img_paths.append(os.path.join(root, file))

    if max_images:
        img_paths = img_paths[:max_images]

    logging.info(f"📊 找到 {len(img_paths)} 张待检测图像")

    # 批量检测
    total_defects = 0
    failed_images = []

    for img_path in tqdm(img_paths, desc="批量像素级检测"):
        try:
            report = defector.detect_pixel_level_defects(img_path, save_dir)
            if report:
                total_defects += report["n_defects"]
        except Exception as e:
            failed_images.append({"path": img_path, "error": str(e)})
            logging.error(f"批量检测失败 {img_path}: {str(e)}")

    # 生成批量报告
    batch_report = {
        "batch_id": time.strftime("%Y%m%d_%H%M%S"),
        "test_directory": test_dir,
        "total_images": len(img_paths),
        "success_images": len(img_paths) - len(failed_images),
        "failed_images": failed_images,
        "total_defects_detected": total_defects,
        "avg_defects_per_image": (
            total_defects / len(img_paths) if len(img_paths) > 0 else 0
        ),
        "config": CONFIG,
    }

    with open(
        os.path.join(save_dir, "batch_detection_report.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(batch_report, f, ensure_ascii=False, indent=2)

    logging.info("📋 批量检测完成，报告已保存")
    logging.info(
        f"📊 统计: 总图像{len(img_paths)}, 成功{len(img_paths)-len(failed_images)}, "
        f"失败{len(failed_images)}, 总瑕疵{total_defects}"
    )

    return batch_report


# ===================== 主程序 =====================
def main():
    """
    主程序：大样本特征学习 + 像素级瑕疵检测
    """
    print("=" * 80)
    print("🚀 布料像素级瑕疵检测系统（适配6000+正常样本）")
    print("=" * 80)

    # 配置路径
    NORMAL_DIR = r"F:\All_dataset\buliao_datasets260107"  # 6000+正常样本目录
    FEATURE_LIB_PATH = "PatchCoreDetection\\feature_lib\\feature_lib_260107_02.npz"

    # 模式选择
    print("\\n请选择操作模式:")
    print("1. 构建大样本特征库（6000+正常样本）")
    print("2. 加载特征库进行单张图像像素级检测")
    print("3. 加载特征库进行批量图像像素级检测")
    print("4. 退出")

    while True:
        mode = input("\\n请输入模式编号 (1/2/3/4): ").strip()

        if mode == "1":
            # 构建特征库
            logging.info("🔨 开始构建大样本特征库")
            try:
                build_large_scale_feature_library(NORMAL_DIR, FEATURE_LIB_PATH)
                print("✅ 特征库构建完成！")
            except Exception as e:
                logging.error(f"特征库构建失败: {str(e)}")
                print(f"❌ 特征库构建失败: {str(e)}")

        elif mode == "2":
            # 单张检测
            try:
                # 加载特征库
                index, feature_lib = load_large_scale_feature_library(FEATURE_LIB_PATH)
                # 创建检测器
                detector = PixelLevelDefectDetector(index, feature_lib)
                # 输入图像路径
                img_path = input("请输入待检测图像路径: ").strip()
                if os.path.exists(img_path):
                    detector.detect_pixel_level_defects(img_path)
                    print("✅ 像素级检测完成！结果已保存至 pixel_level_results 目录")
                else:
                    print(f"❌ 图像文件不存在: {img_path}")
            except Exception as e:
                logging.error(f"单张检测失败: {str(e)}")
                print(f"❌ 检测失败: {str(e)}")

        elif mode == "3":
            # 批量检测
            try:
                # 加载特征库
                index, feature_lib = load_large_scale_feature_library(FEATURE_LIB_PATH)
                # 创建检测器
                detector = PixelLevelDefectDetector(index, feature_lib)
                # 输入测试目录
                test_dir = input("请输入待检测图像目录路径: ").strip()
                if os.path.isdir(test_dir):
                    batch_detect_pixel_level(detector, test_dir)
                    print(
                        "✅ 批量像素级检测完成！结果已保存至 batch_pixel_results 目录"
                    )
                else:
                    print(f"❌ 目录不存在: {test_dir}")
            except Exception as e:
                logging.error(f"批量检测失败: {str(e)}")
                print(f"❌ 批量检测失败: {str(e)}")

        elif mode == "4":
            print("👋 程序退出")
            break

        else:
            print("❌ 无效的模式选择，请输入 1/2/3/4")


if __name__ == "__main__":
    main()
