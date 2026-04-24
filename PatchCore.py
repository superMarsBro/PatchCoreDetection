# 先导入所有核心依赖（按优先级排序）
import torch
import torch.nn as nn
import cv2
import numpy as np
import os
import warnings
from PIL import Image
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from torchvision import transforms
from torchvision.models import convnext_small, ConvNeXt_Small_Weights

# 尝试导入FAISS（加速距离计算），失败则用纯Numpy版本
try:
    import faiss

    FAISS_AVAILABLE = True
    warnings.warn("✅ 已加载FAISS，将使用FAISS加速Coreset采样")
except ImportError:
    FAISS_AVAILABLE = False
    warnings.warn(
        "⚠️ 未安装FAISS，将使用纯Numpy优化版本（速度较慢），建议安装：pip install faiss-cpu"
    )

# 全局配置（必须在torch导入后定义）
IMG_SIZE = (512, 512)  # 统一图像尺寸
PATCH_SIZE = 32  # 特征patch对应像素大小（需和特征提取器匹配）
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_CORESET_SIZE = 5000  # 限制最大采样数（避免采样过多）
REDUCTION_RATIO = 0.1  # 采样比例（可根据需求调整）


# ---------------------- 原有函数（无修改） ----------------------
# 1. 尺寸归一化
def resize_img(img, target_size=IMG_SIZE):
    return cv2.resize(img, target_size, interpolation=cv2.INTER_CUBIC)


# 2. 光照归一化（消除明暗不均）
def normalize_light(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)


# 3. 纹理保留去噪
def denoise_img(img):
    return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)


# 4. PyTorch张量转换
def to_tensor(img):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return transform(img_rgb).unsqueeze(0)  # [1, 3, H, W]


# 完整预处理流程
def preprocess(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"图像读取失败: {img_path}")
    img_resized = resize_img(img)
    img_light = normalize_light(img_resized)
    img_denoised = denoise_img(img_light)
    img_tensor = to_tensor(img_denoised)
    return img_denoised, img_tensor


# PatchCore 特征提取器
class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        weights = ConvNeXt_Small_Weights.DEFAULT
        self.backbone = convnext_small(weights=weights)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x):
        features = self.backbone(x)
        return features


# 初始化特征提取器
extractor = FeatureExtractor().to(DEVICE).eval()


# 加载正常布料路径
def load_normal_paths(normal_dir):
    normal_paths = []
    for root, _, files in os.walk(normal_dir):
        for file in files:
            if file.lower().endswith(("jpg", "jpeg", "png", "bmp")):
                normal_paths.append(os.path.join(root, file))
    return normal_paths


# 提取patch特征
def extract_patch_features(img_paths, extractor):
    all_features = []
    with torch.no_grad():
        for img_path in tqdm(img_paths, desc="提取正常特征"):
            _, img_tensor = preprocess(img_path)
            img_tensor = img_tensor.to(DEVICE)
            feats = extractor(img_tensor)
            feats = feats.permute(0, 2, 3, 1).reshape(-1, 768)
            all_features.append(feats.cpu().numpy())
    all_features = np.concatenate(all_features, axis=0).astype(
        np.float32
    )  # 转float32减少计算量
    return all_features


# ---------------------- 优化后的Coreset采样（核心修改） ----------------------
def coreset_sampling_faiss(features, reduction_ratio=REDUCTION_RATIO):
    """FAISS加速版Coreset采样（推荐）"""
    n_samples, dim = features.shape
    # 限制采样数（避免过多）
    n_coreset = max(1000, min(MAX_CORESET_SIZE, int(n_samples * reduction_ratio)))
    if n_coreset >= n_samples:
        return features  # 无需采样

    # 初始化FAISS索引（欧氏距离）
    index = faiss.IndexFlatL2(dim)
    index.add(features)

    # 初始化采样：随机选第一个点
    selected_idx = [np.random.choice(n_samples)]
    # 初始化最小距离（所有点到已选点的最小距离）
    min_distances, _ = index.search(features[selected_idx], n_samples)
    min_distances = min_distances[0].copy()

    # 批量采样（用tqdm显示进度）
    pbar = tqdm(range(n_coreset - 1), desc="Coreset采样（FAISS加速）")
    for _ in pbar:
        # 选距离最远的点
        next_idx = np.argmax(min_distances)
        selected_idx.append(next_idx)
        # 更新最小距离（距离新选点的距离 vs 现有最小距离）
        new_distances, _ = index.search(features[next_idx : next_idx + 1], n_samples)
        min_distances = np.minimum(min_distances, new_distances[0])
        # 更新进度条（可选）
        pbar.set_postfix(
            {
                "已选点数": len(selected_idx),
                "最大距离": f"{min_distances[next_idx]:.2f}",
            }
        )

    pbar.close()
    return features[selected_idx]


def coreset_sampling_numpy(features, reduction_ratio=REDUCTION_RATIO):
    """纯Numpy优化版Coreset采样（无依赖）"""
    n_samples, dim = features.shape
    n_coreset = max(1000, min(MAX_CORESET_SIZE, int(n_samples * reduction_ratio)))
    if n_coreset >= n_samples:
        return features

    # 预计算特征范数（优化距离计算：||a-b||² = ||a||² + ||b||² - 2a·b）
    feat_norms = np.sum(features**2, axis=1)  # [n_samples,]
    # 初始化采样
    selected_idx = [np.random.choice(n_samples)]
    # 初始化最小距离（用平方距离避免开根号，加速计算）
    selected_feat = features[selected_idx[0]]
    distances = (
        feat_norms + feat_norms[selected_idx[0]] - 2 * features @ selected_feat.T
    )
    distances = np.maximum(distances, 0)  # 避免数值误差导致负数

    # 批量采样
    pbar = tqdm(range(n_coreset - 1), desc="Coreset采样（Numpy优化）")
    for _ in pbar:
        next_idx = np.argmax(distances)
        selected_idx.append(next_idx)
        # 更新最小距离（平方距离）
        new_feat = features[next_idx]
        new_distances = feat_norms + feat_norms[next_idx] - 2 * features @ new_feat.T
        new_distances = np.maximum(new_distances, 0)
        distances = np.minimum(distances, new_distances)
        pbar.set_postfix(
            {
                "已选点数": len(selected_idx),
                "最大距离": f"{np.sqrt(distances[next_idx]):.2f}",
            }
        )

    pbar.close()
    return features[selected_idx]


# 统一采样接口（自动选择FAISS/Numpy）
def coreset_sampling(features, reduction_ratio=REDUCTION_RATIO):
    if FAISS_AVAILABLE:
        return coreset_sampling_faiss(features, reduction_ratio)
    else:
        return coreset_sampling_numpy(features, reduction_ratio)


# ---------------------- 原有函数（无修改） ----------------------
def build_feature_library(
    normal_dir, save_path="PatchCoreDetection\\feature_lib_2.npz"
):
    """构建并保存正常特征库"""
    normal_paths = load_normal_paths(normal_dir)
    if len(normal_paths) < 50:
        raise ValueError("正常样本数量过少（至少50张）")

    # 提取patch特征
    raw_features = extract_patch_features(normal_paths, extractor)
    print(f"📌 原始特征数：{len(raw_features)}个patch")

    # Coreset采样（优化后的版本）
    core_features = coreset_sampling(raw_features)

    # 构建KNN索引
    knn = NearestNeighbors(
        n_neighbors=9, metric="euclidean", n_jobs=-1
    )  # n_jobs=-1用多线程
    knn.fit(core_features)

    # 保存特征库
    np.savez(save_path, features=core_features, knn_data=knn._fit_X)
    print(f"✅ 特征库构建完成：{len(core_features)}个patch，保存至{save_path}")
    return knn, core_features


def load_feature_library(load_path="PatchCoreDetection\\feature_lib_2.npz"):
    """加载已保存的特征库并重建KNN索引"""
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"特征库文件不存在：{load_path}")

    # 加载npz文件
    data = np.load(load_path)
    core_features = data["features"]
    # 重建KNN索引（保持和构建时一致的参数）
    knn = NearestNeighbors(n_neighbors=9, metric="euclidean", n_jobs=-1)
    knn.fit(core_features)

    print(f"✅ 特征库加载完成：{len(core_features)}个patch，来自{load_path}")
    return knn, core_features


# 计算异常分数
def compute_anomaly_score(img_tensor, extractor, knn):
    with torch.no_grad():
        feats = extractor(img_tensor.to(DEVICE))
        feats = feats.permute(0, 2, 3, 1).reshape(-1, 768)
        feats_np = feats.cpu().numpy().astype(np.float32)

    distances, _ = knn.kneighbors(feats_np)
    patch_scores = np.mean(distances, axis=1)
    return patch_scores


# 生成热力图
def generate_heatmap(patch_scores, img_shape=IMG_SIZE):
    patch_h, patch_w = 16, 16
    score_map = patch_scores.reshape(patch_h, patch_w)
    heatmap = cv2.resize(score_map, img_shape, interpolation=cv2.INTER_CUBIC)
    return heatmap


# 后处理
def post_process(heatmap, img_ori, threshold=None):
    if threshold is None:
        _, binary_mask = cv2.threshold(
            (heatmap / heatmap.max() * 255).astype(np.uint8),
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
    else:
        binary_mask = (heatmap > threshold).astype(np.uint8) * 255

    gray = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)
    local_var = cv2.GaussianBlur(gray, (15, 15), 0)
    binary_mask = cv2.bitwise_and(
        binary_mask, binary_mask, mask=(local_var < 20).astype(np.uint8)
    )

    kernel1 = np.ones((3, 3), np.uint8)
    kernel2 = np.ones((5, 5), np.uint8)
    binary_mask = cv2.erode(binary_mask, kernel1, iterations=1)
    binary_mask = cv2.dilate(binary_mask, kernel2, iterations=2)

    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 10]

    result_img = img_ori.copy()
    defect_boxes = []
    for cnt in valid_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        defect_boxes.append((x, y, x + w, y + h))
        cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        area = cv2.contourArea(cnt)
        if area < 100:
            label = "污点"
        elif w / h > 3 or h / w > 3:
            label = "跳线/划痕"
        else:
            label = "瑕疵/破洞"
        cv2.putText(
            result_img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1
        )

    clean_mask = np.zeros_like(binary_mask)
    cv2.drawContours(clean_mask, valid_contours, -1, 255, -1)

    return result_img, clean_mask, defect_boxes


# 缺陷检测主函数
def detect_defect(img_path, knn, save_dir="results"):
    img_ori, img_tensor = preprocess(img_path)
    patch_scores = compute_anomaly_score(img_tensor, extractor, knn)
    heatmap = generate_heatmap(patch_scores)
    result_img, defect_mask, defect_boxes = post_process(heatmap, img_ori)

    os.makedirs(save_dir, exist_ok=True)
    img_name = os.path.basename(img_path)
    cv2.imwrite(os.path.join(save_dir, f"result_{img_name}"), result_img)
    cv2.imwrite(os.path.join(save_dir, f"mask_{img_name}"), defect_mask)

    print(f"✅ 检测完成：{len(defect_boxes)}个缺陷区域")
    for i, (x1, y1, x2, y2) in enumerate(defect_boxes):
        print(f"   缺陷{i+1}：位置({x1},{y1})-({x2},{y2})，面积{(x2-x1)*(y2-y1)}像素")

    return result_img, defect_mask, defect_boxes


if __name__ == "__main__":
    # ====================== 模式选择 ======================
    # 模式1：构建特征库（仅首次运行）
    # knn, feature_lib = build_feature_library(
    #     normal_dir=r"F:\All_dataset\\normal_buliao251208_5"
    # )

    # # 模式2：加载已构建的特征库（日常检测用）
    knn, feature_lib = load_feature_library(
        load_path="PatchCoreDetection\\feature_lib_2.npz"  # 特征库路径
    )

    # ====================== 缺陷检测 ======================
    # 单张图片检测示例
    test_img_path = r"F:\All_dataset\\fake_negative\\000000_251205_001.png"  # 替换为你的缺陷图片路径
    result_img, defect_mask, defect_boxes = detect_defect(
        img_path=test_img_path,
        knn=knn,
        save_dir="PatchCoreDetection\\detection_results",  # 检测结果保存路径
    )

    # 批量图片检测示例（可选）
    # test_dir = r"F:\All_dataset\defect_buliao"  # 缺陷图片文件夹
    # for root, _, files in os.walk(test_dir):
    #     for file in files:
    #         if file.lower().endswith(("jpg", "jpeg", "png", "bmp")):
    #             test_img_path = os.path.join(root, file)
    #             detect_defect(img_path=test_img_path, knn=knn, save_dir="PatchCoreDetection\\detection_results")
