import torch
import torch.nn as nn
import cv2
import numpy as np
import os
import warnings
from PIL import Image
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN  # 新增：用于异常点过滤
from torchvision import transforms
from torchvision.models import convnext_small, ConvNeXt_Small_Weights
import time
import json
from scipy import ndimage

# 尝试导入FAISS
try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    warnings.warn("未安装FAISS，将使用scikit-learn KNN")

# ===================== 优化后的全局配置 =====================
IMG_SIZE = (512, 512)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 关键参数 - 根据你的布料类型调整
CONFIG = {
    # 特征库构建参数
    "n_neighbors": 20,  # 增加邻居数（5→9），减少误报
    "min_normal_samples": 30,  # 最小正常样本数（降低要求）
    "coreset_ratio": 0.05,  # Coreset采样比例（降低计算量）
    "max_coreset_size": 3000,  # 最大核心特征数
    # 预处理参数
    "clahe_clip_limit": 1.5,  # CLAHE对比度限制（降低）
    "denoise_strength": 7,  # 去噪强度
    # 检测参数
    "threshold_std_multiplier": 3.5,  # 阈值倍数（增加减少误报）
    "min_defect_area": 30,  # 最小缺陷面积
    "texture_variance_threshold": 25,  # 纹理方差阈值
    "use_adaptive_threshold": True,  # 使用自适应阈值
    # 后处理参数
    "morphology_kernel_size": 5,  # 形态学核大小
    "area_filtering_enabled": True,  # 面积过滤
    "contour_smoothing": True,  # 轮廓平滑
}

print(f"🔧 使用配置: {json.dumps(CONFIG, indent=2, ensure_ascii=False)}")


# ===================== 改进的预处理 =====================
def improved_preprocess(img_path):
    """改进的预处理流程：减少过度增强"""
    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"图像读取失败: {img_path}")

    # 1. 尺寸归一化
    img_resized = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)  # 改为AREA

    # 2. 温和的光照归一化
    lab = cv2.cvtColor(img_resized, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)  # 转换到LAB颜色空间

    # 使用更温和的CLAHE参数
    clahe = cv2.createCLAHE(clipLimit=CONFIG["clahe_clip_limit"], tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    lab_enhanced = cv2.merge((l_enhanced, a, b))
    img_normalized = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    # 3. 保留纹理的去噪
    img_denoised = cv2.fastNlMeansDenoisingColored(
        img_normalized,
        None,
        h=CONFIG["denoise_strength"],
        hColor=CONFIG["denoise_strength"],
        templateWindowSize=7,
        searchWindowSize=21,
    )

    # 4. 转换为张量
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    img_rgb = cv2.cvtColor(img_denoised, cv2.COLOR_BGR2RGB)
    img_tensor = transform(img_rgb).unsqueeze(0)

    return img_denoised, img_tensor


# ===================== 特征提取器（保持不变） =====================
# 基于ConvNeXt-Small的预训练模型
# 输出：16×16×768的特征图（对应原图32×32个区域）
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


extractor = FeatureExtractor().to(DEVICE).eval()


# ===================== 改进的特征库构建 =====================
def build_optimized_feature_library(
    normal_dir, save_path="feature_lib_optimized_3.npz"
):
    """构建优化后的特征库：加入异常点过滤"""
    print("🚀 构建优化特征库...")

    # 收集正常图像路径
    normal_paths = []
    for root, _, files in os.walk(normal_dir):
        for file in files:
            if file.lower().endswith(("jpg", "jpeg", "png", "bmp")):
                normal_paths.append(os.path.join(root, file))

    if len(normal_paths) < CONFIG["min_normal_samples"]:
        raise ValueError(
            f"正常样本数量不足: {len(normal_paths)} (< {CONFIG['min_normal_samples']})"
        )

    print(f"📁 找到 {len(normal_paths)} 张正常图像")

    # 提取特征
    all_features = []
    with torch.no_grad():
        for img_path in tqdm(normal_paths, desc="提取正常特征"):
            try:
                _, img_tensor = improved_preprocess(img_path)
                img_tensor = img_tensor.to(DEVICE)
                feats = extractor(img_tensor)
                feats = feats.permute(0, 2, 3, 1).reshape(-1, 768)
                all_features.append(feats.cpu().numpy())
            except Exception as e:
                print(f"⚠️ 跳过 {img_path}: {e}")
                continue

    if not all_features:
        raise ValueError("无有效特征可提取")

    all_features = np.concatenate(all_features, axis=0).astype(np.float32)
    print(f"📊 原始特征数: {all_features.shape[0]}")

    # # 步骤1: 使用DBSCAN过滤异常点
    # print("🔍 使用DBSCAN过滤异常点...")
    # dbscan = DBSCAN(eps=0.5, min_samples=10, metric="euclidean", n_jobs=-1)
    # labels = dbscan.fit_predict(all_features)

    # # 保留核心点（非异常点）
    # core_mask = labels != -1
    # filtered_features = all_features[core_mask]
    # print(
    #     f"✅ 过滤后特征数: {filtered_features.shape[0]} (移除{np.sum(labels==-1)}个异常点)"
    # )

    # ✅ 简单快速的异常过滤：基于特征范数,直接去除了DBSCAN
    print("🔍 使用快速异常过滤...")
    feature_norms = np.linalg.norm(all_features, axis=1)
    norm_mean = np.mean(feature_norms)
    norm_std = np.std(feature_norms)

    # 过滤掉范数异常的特征（太极端的值）
    upper_bound = norm_mean + 2.5 * norm_std
    lower_bound = norm_mean - 2.5 * norm_std

    core_mask = (feature_norms >= lower_bound) & (feature_norms <= upper_bound)
    filtered_features = all_features[core_mask]
    print(
        f"✅ 过滤后特征数: {filtered_features.shape[0]} (移除{np.sum(~core_mask)}个异常点)"
    )

    # 步骤2: 改进的Coreset采样
    print("🎯 执行优化Coreset采样...")
    sampled_features = improved_coreset_sampling(filtered_features)

    # 步骤3: 构建KNN索引
    print("🔧 构建KNN索引...")
    knn = NearestNeighbors(
        n_neighbors=CONFIG["n_neighbors"], metric="euclidean", n_jobs=-1
    )
    knn.fit(sampled_features)

    # 保存优化后的特征库
    np.savez(
        save_path,
        features=sampled_features,
        knn_data=knn._fit_X,
        config=CONFIG,
        n_samples=len(normal_paths),
    )

    print(f"💾 优化特征库保存至: {save_path}")
    print(f"📈 最终特征数: {sampled_features.shape[0]}")

    return knn, sampled_features


def improved_coreset_sampling(features, method="random"):
    """改进的采样策略"""
    n_samples = features.shape[0]
    target_size = min(
        CONFIG["max_coreset_size"], int(n_samples * CONFIG["coreset_ratio"])
    )

    if method == "random":
        # 简单随机采样（速度快）
        idx = np.random.choice(n_samples, target_size, replace=False)
        return features[idx]

    elif method == "kmeans":
        # K-Means聚类采样（质量高）
        from sklearn.cluster import MiniBatchKMeans

        n_clusters = min(500, target_size // 2)
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters, batch_size=1000, n_init=3, random_state=42
        )

        labels = kmeans.fit_predict(features)

        # 从每个聚类中采样
        sampled_indices = []
        samples_per_cluster = max(1, target_size // n_clusters)

        for cluster_id in range(n_clusters):
            cluster_indices = np.where(labels == cluster_id)[0]
            if len(cluster_indices) > 0:
                if len(cluster_indices) <= samples_per_cluster:
                    sampled_indices.extend(cluster_indices)
                else:
                    selected = np.random.choice(
                        cluster_indices, samples_per_cluster, replace=False
                    )
                    sampled_indices.extend(selected)

        return features[sampled_indices]

    else:
        raise ValueError(f"未知采样方法: {method}")


# ===================== 加载特征库 =====================
def load_optimized_feature_library(load_path="feature_lib_optimized_3.npz"):
    """加载优化后的特征库"""
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"特征库文件不存在: {load_path}")

    data = np.load(load_path, allow_pickle=True)
    core_features = data["features"]

    # 重建KNN索引
    knn = NearestNeighbors(
        n_neighbors=CONFIG["n_neighbors"], metric="euclidean", n_jobs=-1
    )
    knn.fit(core_features)

    print(f"✅ 加载优化特征库: {len(core_features)}个特征")
    if "config" in data:
        print(f"📋 配置参数: {data['config'].item()}")

    return knn, core_features


# ===================== 改进的缺陷检测 =====================
class AdvancedDefectDetector:
    """高级缺陷检测器：减少误报"""

    def __init__(self, knn_model, feature_lib):
        self.knn = knn_model
        self.feature_lib = feature_lib

    def compute_robust_anomaly_scores(self, img_tensor):
        """计算鲁棒的异常分数"""
        with torch.no_grad():
            img_tensor = img_tensor.to(DEVICE)
            feats = extractor(img_tensor)
            feats = feats.permute(0, 2, 3, 1).reshape(-1, 768)
            feats_np = feats.cpu().numpy().astype(np.float32)

        # 使用多个邻居距离的中位数（比平均值更鲁棒）
        distances, _ = self.knn.kneighbors(feats_np)

        # 策略1: 使用中位数距离
        median_distances = np.median(distances, axis=1)

        # 策略2: 加权平均（给近距离更高权重）
        weights = 1.0 / (distances + 1e-6)
        weights = weights / weights.sum(axis=1, keepdims=True)
        weighted_avg = np.sum(distances * weights, axis=1)

        # 策略3: 取前k/2个最小距离的平均
        k_half = distances.shape[1] // 2
        sorted_distances = np.sort(distances, axis=1)
        avg_smallest = np.mean(sorted_distances[:, :k_half], axis=1)

        # 综合多个策略
        final_scores = 0.4 * median_distances + 0.3 * weighted_avg + 0.3 * avg_smallest

        return final_scores

    def generate_improved_heatmap(self, patch_scores, img_shape=IMG_SIZE):
        """生成改进的热力图"""
        patch_h, patch_w = 16, 16
        score_map = patch_scores.reshape(patch_h, patch_w)

        # 使用各向异性高斯滤波（保留边缘）
        heatmap = cv2.GaussianBlur(score_map, (3, 3), sigmaX=1.0, sigmaY=1.0)

        # 上采样
        heatmap = cv2.resize(heatmap, img_shape, interpolation=cv2.INTER_CUBIC)

        return heatmap

    def adaptive_thresholding(self, heatmap):
        """自适应阈值策略"""
        if CONFIG["use_adaptive_threshold"]:
            # 方法1: 基于局部统计的自适应阈值
            mean_val = np.mean(heatmap)
            std_val = np.std(heatmap)
            threshold = mean_val + CONFIG["threshold_std_multiplier"] * std_val

            # 方法2: 分块自适应阈值（处理光照不均）
            h, w = heatmap.shape
            block_size = 64
            adaptive_map = np.zeros_like(heatmap)

            for i in range(0, h, block_size):
                for j in range(0, w, block_size):
                    block = heatmap[
                        i : min(i + block_size, h), j : min(j + block_size, w)
                    ]
                    if block.size > 0:
                        block_mean = np.mean(block)
                        block_std = np.std(block)
                        block_threshold = block_mean + 2.5 * block_std
                        adaptive_map[
                            i : min(i + block_size, h), j : min(j + block_size, w)
                        ] = block_threshold

            # 取两种方法的较小值作为阈值
            final_threshold = np.minimum(
                threshold, np.mean(adaptive_map[adaptive_map > 0])
            )
        else:
            # 固定阈值
            final_threshold = np.percentile(heatmap, 97)  # 取95%分位数

        return final_threshold

    def post_process_with_context(self, heatmap, original_img):
        """上下文感知的后处理"""
        # 1. 自适应阈值
        threshold = self.adaptive_thresholding(heatmap)
        binary_mask = (heatmap > threshold).astype(np.uint8) * 255

        # 2. 纹理一致性分析
        gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

        # 计算局部纹理方差
        local_var = cv2.GaussianBlur(gray, (15, 15), 0)
        var_mask = (local_var < CONFIG["texture_variance_threshold"]).astype(np.uint8)

        # 3. 梯度一致性分析（排除边缘区域）
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        grad_mask = (gradient_magnitude < 50).astype(np.uint8)

        # 4. 综合掩码（只有低纹理方差且非边缘的区域才可能是缺陷）
        combined_mask = cv2.bitwise_and(var_mask, grad_mask)
        binary_mask = cv2.bitwise_and(binary_mask, binary_mask, mask=combined_mask)

        # 5. 形态学操作
        kernel_size = CONFIG["morphology_kernel_size"]
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )

        # 先闭运算连接邻近区域，再开运算去除小噪声
        binary_mask = cv2.morphologyEx(
            binary_mask, cv2.MORPH_CLOSE, kernel, iterations=1
        )
        binary_mask = cv2.morphologyEx(
            binary_mask, cv2.MORPH_OPEN, kernel, iterations=1
        )

        # 6. 连通域分析
        contours, _ = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # 7. 多级过滤
        valid_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)

            # 面积过滤
            if area < CONFIG["min_defect_area"]:
                continue

            # 形状过滤（排除太细长或太圆的区域）
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue

            circularity = 4 * np.pi * area / (perimeter**2)
            if circularity > 0.9:  # 太圆可能是纹理点
                continue

            # 紧凑度过滤
            x, y, w, h = cv2.boundingRect(cnt)
            rect_area = w * h
            if rect_area == 0:
                continue

            compactness = area / rect_area
            if compactness < 0.3:  # 太分散可能是噪声
                continue

            valid_contours.append(cnt)

        # 8. 生成结果
        result_img = original_img.copy()
        defect_boxes = []
        defect_info = []

        for i, cnt in enumerate(valid_contours):
            x, y, w, h = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)

            defect_boxes.append((x, y, x + w, y + h))

            # 缺陷分类
            if area < 100:
                defect_type = "小污点"
                color = (0, 0, 255)  # 红色
            elif w / h > 3 or h / w > 3:
                defect_type = "划痕"
                color = (0, 255, 255)  # 黄色
            elif area > 500:
                defect_type = "大瑕疵"
                color = (255, 0, 0)  # 蓝色
            else:
                defect_type = "瑕疵"
                color = (0, 255, 0)  # 绿色

            defect_info.append(
                {
                    "id": i + 1,
                    "bbox": (x, y, w, h),
                    "area": area,
                    "type": defect_type,
                    "confidence": np.mean(heatmap[y : y + h, x : x + w]),
                }
            )

            # 绘制
            cv2.rectangle(result_img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                result_img,
                f"{defect_type}({area})",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

        # 重新生成掩码
        clean_mask = np.zeros_like(binary_mask)
        cv2.drawContours(clean_mask, valid_contours, -1, 255, -1)

        return result_img, clean_mask, defect_boxes, defect_info, heatmap, threshold

    def detect_with_confidence(self, img_path, save_dir="optimized_results"):
        """完整的缺陷检测流程"""
        print(f"\n🔍 检测图像: {os.path.basename(img_path)}")

        # 1. 预处理
        try:
            img_processed, img_tensor = improved_preprocess(img_path)
            img_original = cv2.imread(img_path)
            img_original = cv2.resize(img_original, IMG_SIZE)
        except Exception as e:
            print(f"❌ 预处理失败: {e}")
            return None

        # 2. 计算异常分数
        start_time = time.time()
        patch_scores = self.compute_robust_anomaly_scores(img_tensor)

        # 3. 生成热力图
        heatmap = self.generate_improved_heatmap(patch_scores)

        # 4. 后处理
        result_img, defect_mask, defect_boxes, defect_info, heatmap, threshold = (
            self.post_process_with_context(heatmap, img_processed)
        )

        detection_time = time.time() - start_time

        # 5. 保存结果
        os.makedirs(save_dir, exist_ok=True)
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        # 保存图像
        cv2.imwrite(os.path.join(save_dir, f"{img_name}_result.jpg"), result_img)
        cv2.imwrite(os.path.join(save_dir, f"{img_name}_mask.jpg"), defect_mask)

        # 保存热力图
        heatmap_normalized = (heatmap - heatmap.min()) / (
            heatmap.max() - heatmap.min() + 1e-6
        )
        heatmap_colored = cv2.applyColorMap(
            (heatmap_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        cv2.imwrite(os.path.join(save_dir, f"{img_name}_heatmap.jpg"), heatmap_colored)

        # 保存报告
        self.save_detection_report(
            img_path, defect_info, threshold, detection_time, save_dir
        )

        # 6. 输出结果
        print(f"✅ 检测完成:")
        print(f"   检测阈值: {threshold:.4f}")
        print(f"   缺陷数量: {len(defect_info)}")
        print(f"   检测时间: {detection_time:.3f}秒")

        for info in defect_info:
            print(
                f"   缺陷{info['id']}: {info['type']}, 面积{info['area']}像素, "
                f"置信度{info['confidence']:.3f}"
            )

        return {
            "result_img": result_img,
            "defect_mask": defect_mask,
            "heatmap": heatmap,
            "defect_info": defect_info,
            "detection_time": detection_time,
            "threshold": threshold,
        }

    def save_detection_report(
        self, img_path, defect_info, threshold, detection_time, save_dir
    ):
        """保存检测报告"""
        report_path = os.path.join(save_dir, "detection_report.txt")

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("布料缺陷检测报告（优化版）\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"检测时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"图像文件: {os.path.basename(img_path)}\n")
            f.write(f"检测耗时: {detection_time:.3f}秒\n")
            f.write(f"检测阈值: {threshold:.4f}\n")
            f.write(f"缺陷总数: {len(defect_info)}\n\n")

            if defect_info:
                f.write("缺陷详情:\n")
                f.write("-" * 60 + "\n")
                for info in defect_info:
                    f.write(f"缺陷{info['id']}:\n")
                    f.write(f"  类型: {info['type']}\n")
                    f.write(f"  位置: {info['bbox']}\n")
                    f.write(f"  面积: {info['area']}像素\n")
                    f.write(f"  置信度: {info['confidence']:.3f}\n")
                    f.write("-" * 30 + "\n")
            else:
                f.write("✅ 未检测到缺陷\n")

        print(f"📄 检测报告已保存: {report_path}")


# ===================== 性能评估工具 =====================
def evaluate_detector(detector, test_images, ground_truth=None):
    """评估检测器性能"""
    results = []

    for img_path in test_images:
        print(f"\n📊 评估: {os.path.basename(img_path)}")

        # 检测
        result = detector.detect_with_confidence(img_path, "evaluation_results")

        if result:
            # 计算性能指标
            defect_count = len(result["defect_info"])

            if ground_truth and img_path in ground_truth:
                gt_defects = ground_truth[img_path]
                # 计算精确率、召回率等
                # 这里需要你有标注数据
                pass

            results.append(
                {
                    "image": os.path.basename(img_path),
                    "defect_count": defect_count,
                    "detection_time": result["detection_time"],
                    "threshold": result["threshold"],
                }
            )

    # 生成评估报告
    print("\n" + "=" * 60)
    print("性能评估报告")
    print("=" * 60)

    total_images = len(results)
    total_defects = sum(r["defect_count"] for r in results)
    avg_time = np.mean([r["detection_time"] for r in results])

    print(f"总测试图像: {total_images}")
    print(f"总检测缺陷: {total_defects}")
    print(f"平均检测时间: {avg_time:.3f}秒/张")
    print(f"平均每图缺陷数: {total_defects/total_images:.2f}")


# ===================== 主程序 =====================
def main():
    """主函数：使用优化后的方法"""
    print("=" * 60)
    print("🚀 布料缺陷检测系统 - 优化版（减少误报）")
    print("=" * 60)

    # 模式选择
    print("\n请选择模式:")
    print("1. 构建新的优化特征库")
    print("2. 加载已有特征库进行检测")
    print("3. 批量检测目录")

    mode = input("请输入选择 (1/2/3): ").strip()

    # 路径配置
    NORMAL_DIR = r"F:\All_dataset\\normal_buliao251208_5"
    FEATURE_LIB_PATH = "PatchCoreDetection/feature_lib_3.npz"

    if mode == "1":
        # 构建优化特征库
        print("\n🔨 构建优化特征库...")
        knn, feature_lib = build_optimized_feature_library(
            normal_dir=NORMAL_DIR, save_path=FEATURE_LIB_PATH
        )

        # 测试一张图像
        test_img = input("输入测试图像路径: ").strip()
        if os.path.exists(test_img):
            detector = AdvancedDefectDetector(knn, feature_lib)
            detector.detect_with_confidence(test_img, "test_results")

    elif mode == "2":
        # 加载特征库
        print("\n📦 加载特征库...")
        knn, feature_lib = load_optimized_feature_library(FEATURE_LIB_PATH)

        # 创建检测器
        detector = AdvancedDefectDetector(knn, feature_lib)

        # 单张检测
        test_img = input("输入测试图像路径 (或直接回车使用默认): ").strip()
        if not test_img:
            test_img = r"F:\All_dataset\\fake_negative\\000000_251205_001.png"

        if os.path.exists(test_img):
            detector.detect_with_confidence(test_img, "detection_results")
        else:
            print(f"❌ 图像不存在: {test_img}")

    elif mode == "3":
        # 批量检测
        print("\n📁 批量检测模式")
        knn, feature_lib = load_optimized_feature_library(FEATURE_LIB_PATH)
        detector = AdvancedDefectDetector(knn, feature_lib)

        test_dir = input("输入测试目录路径: ").strip()
        if os.path.isdir(test_dir):
            # 收集图像
            img_paths = []
            for root, _, files in os.walk(test_dir):
                for file in files:
                    if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                        img_paths.append(os.path.join(root, file))

            print(f"找到 {len(img_paths)} 张图像")

            # 批量处理
            for img_path in tqdm(img_paths[:20], desc="批量检测"):  # 限制前20张
                try:
                    detector.detect_with_confidence(img_path, "batch_results")
                except Exception as e:
                    print(f"⚠️ 检测失败 {img_path}: {e}")

        else:
            print(f"❌ 目录不存在: {test_dir}")

    else:
        print("❌ 无效选择")


if __name__ == "__main__":
    main()
