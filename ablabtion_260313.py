import torch
import torch.nn as nn
import cv2
import numpy as np
import os
import time
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models.feature_extraction import create_feature_extractor
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
)
import faiss
import warnings

warnings.filterwarnings("ignore")

# ===================== 1. 全局配置 =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_DIR = r"F:\All_dataset\\long_buliao_datasets_260107_cutbackground"
TEST_GOOD_DIR = r"F:\All_dataset\\260309buliao_datasets\\good"
TEST_BAD_DIR = r"F:\All_dataset\\260309buliao_datasets\bad"

RESULTS_DIR = "Ablation_Results_Strict_260313"
os.makedirs(RESULTS_DIR, exist_ok=True)
LIB_SAVE_DIR = os.path.join(RESULTS_DIR, "feature_libs")
os.makedirs(LIB_SAVE_DIR, exist_ok=True)

CONFIG = {
    "input_size": (384, 1024),
    "batch_extract_size": 8,
    "max_coreset_size": 100000,
    "n_neighbors": 9,
    "nprobe": 5,
}


# ===================== 2. 严格对齐原版的预处理 =====================
def robust_preprocess_ablation(img_path, use_clahe):
    try:
        img_original = cv2.imread(img_path)
        if img_original is None:
            return None
        target_h, target_w = CONFIG["input_size"]
        img_resized = cv2.resize(
            img_original, (target_w, target_h), interpolation=cv2.INTER_LINEAR
        )

        # 严格遵循 Test 4 (Baseline) 中注释掉 LAB 的逻辑
        if use_clahe:
            lab = cv2.cvtColor(img_resized, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            img_enhanced = cv2.merge((l, a, b))
            img_enhanced = cv2.cvtColor(img_enhanced, cv2.COLOR_LAB2BGR)
        else:
            img_enhanced = img_resized

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
        return img_tensor
    except Exception:
        return None


# ===================== 3. 严格对齐原版的特征提取器 =====================
class StrictFeatureExtractor(nn.Module):
    def __init__(self, layers_config, pool_size):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT
        base_model = resnet18(weights=weights)

        self.layers_config = layers_config
        if self.layers_config == "deep":
            self.feature_extractor = create_feature_extractor(
                base_model, return_nodes={"layer2": "s2", "layer3": "s3"}
            )
        else:
            self.feature_extractor = create_feature_extractor(
                base_model, return_nodes={"layer1": "s1", "layer2": "s2"}
            )

        padding = pool_size // 2
        self.avg_pool = nn.AvgPool2d(pool_size, stride=1, padding=padding)

        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.feature_extractor.eval()
        self.fh = 0
        self.fw = 0

    def forward(self, x):
        outputs = self.feature_extractor(x)
        # 严格遵循原版的特征拼接逻辑
        if self.layers_config == "deep":
            s3_up = torch.nn.functional.interpolate(
                outputs["s3"], size=outputs["s2"].shape[-2:], mode="bilinear"
            )
            features = torch.cat([outputs["s2"], s3_up], dim=1)
        else:
            s2_up = torch.nn.functional.interpolate(
                outputs["s2"], size=outputs["s1"].shape[-2:], mode="bilinear"
            )
            features = torch.cat([outputs["s1"], s2_up], dim=1)

        features = self.avg_pool(features)
        self.fh, self.fw = features.shape[2], features.shape[3]
        B, C, H, W = features.shape
        return features.permute(0, 2, 3, 1).reshape(B, H * W, C)


# ===================== 4. 严格对齐原版的核心实验类 =====================
class StrictAblationDetector:
    def __init__(self, name, use_clahe, layers_config, pool_size, blur_size):
        self.name = name
        self.use_clahe = use_clahe
        self.blur_size = blur_size
        self.extractor = StrictFeatureExtractor(layers_config, pool_size).to(DEVICE)
        self.index = None

    def fit(self, train_img_paths, save_dir):
        safe_name = (
            self.name.replace(" ", "_")
            .replace("+", "_")
            .replace("/", "_")
            .replace("&", "_")
        )
        index_path = os.path.join(save_dir, f"{safe_name}.ivf_index")

        if os.path.exists(index_path):
            print(f"🔄 [{self.name}] 加载已有特征库: {index_path}")
            self.index = faiss.read_index(index_path)
            self.index.nprobe = CONFIG["nprobe"]
            if DEVICE.type == "cuda" and hasattr(faiss, "StandardGpuResources"):
                try:
                    res = faiss.StandardGpuResources()
                    self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                except:
                    pass
            return

        features_list = []
        TARGET_MEM = 1500000
        # 【关键修复】使用原版固定计算的 est_patches 逻辑，保障采样率一致
        est_patches = len(train_img_paths) * (
            CONFIG["input_size"][0] // 4 * CONFIG["input_size"][1] // 4
        )
        ratio = min(1.0, TARGET_MEM / est_patches)

        self.extractor.eval()
        for i in tqdm(
            range(0, len(train_img_paths), CONFIG["batch_extract_size"]),
            desc=f"[{self.name}] 正在建库",
        ):
            batch = []
            for p in train_img_paths[i : i + CONFIG["batch_extract_size"]]:
                tensor = robust_preprocess_ablation(p, self.use_clahe)
                if tensor is not None:
                    batch.append(tensor)
            if not batch:
                continue

            with torch.no_grad():
                feat = self.extractor(torch.cat(batch).to(DEVICE)).cpu().numpy()

            for f in feat:
                n = f.shape[0]
                if ratio < 1.0:
                    idx = np.random.choice(n, int(n * ratio), replace=False)
                    features_list.append(f[idx])
                else:
                    features_list.append(f)

        all_features = np.concatenate(features_list, axis=0)
        # 【关键修复】使用原版 numpy float64 的归一化逻辑
        all_features = all_features / (
            np.linalg.norm(all_features, axis=1, keepdims=True) + 1e-8
        )

        if all_features.shape[0] > CONFIG["max_coreset_size"]:
            from sklearn.cluster import MiniBatchKMeans

            kmeans = MiniBatchKMeans(
                n_clusters=CONFIG["max_coreset_size"] // 10,
                batch_size=4096,
                n_init="auto",
            ).fit(all_features)
            core_features = kmeans.cluster_centers_
        else:
            core_features = all_features

        # 【关键修复】使用原版 IndexIVFFlat
        d = core_features.shape[1]
        nlist = int(8 * math.sqrt(core_features.shape[0]))
        nlist = max(32, min(nlist, 2048))
        quantizer = faiss.IndexFlatL2(d)
        self.index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
        self.index.train(core_features)
        self.index.add(core_features)
        self.index.nprobe = CONFIG["nprobe"]

        faiss.write_index(self.index, index_path)

        if DEVICE.type == "cuda" and hasattr(faiss, "StandardGpuResources"):
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
            except:
                pass

    def predict(self, img_path):
        tensor = robust_preprocess_ablation(img_path, self.use_clahe)
        if tensor is None:
            return 0.0

        self.extractor.eval()
        with torch.no_grad():
            feat = self.extractor(tensor.to(DEVICE))
            feat = feat.cpu().numpy().astype(np.float32)[0]
            fh, fw = self.extractor.fh, self.extractor.fw

        # 【关键修复】原版 numpy float32 归一化逻辑
        norm = np.linalg.norm(feat, axis=1, keepdims=True) + 1e-8
        feat = feat / norm

        distances, _ = self.index.search(feat, CONFIG["n_neighbors"])
        score_map = np.mean(distances, axis=1).reshape(fh, fw)
        score_map = cv2.GaussianBlur(score_map, (self.blur_size, self.blur_size), 0)

        # 【极其关键修复】必须在这里 resize 回原图尺寸再求 max！
        h, w = CONFIG["input_size"]
        heatmap = cv2.resize(score_map, (w, h), interpolation=cv2.INTER_LINEAR)
        return np.max(heatmap)


# ===================== 5. 主控制流 =====================
def run_ablation_experiment():
    train_imgs = [
        os.path.join(TRAIN_DIR, f)
        for f in os.listdir(TRAIN_DIR)
        if f.endswith((".jpg", ".png"))
    ]
    good_imgs = [
        os.path.join(TEST_GOOD_DIR, f)
        for f in os.listdir(TEST_GOOD_DIR)
        if f.endswith((".jpg", ".png"))
    ]
    bad_imgs = [
        os.path.join(TEST_BAD_DIR, f)
        for f in os.listdir(TEST_BAD_DIR)
        if f.endswith((".jpg", ".png"))
    ]

    test_imgs = good_imgs + bad_imgs
    test_labels = [0] * len(good_imgs) + [1] * len(bad_imgs)

    # 严格对齐你的 4 份代码配置
    models_to_test = [
        StrictAblationDetector(
            "Test 1 (Baseline)",
            use_clahe=False,
            layers_config="shallow",
            pool_size=3,
            blur_size=3,
        ),
        StrictAblationDetector(
            "Test 2 (+ CLAHE)",
            use_clahe=True,
            layers_config="shallow",
            pool_size=3,
            blur_size=3,
        ),
        StrictAblationDetector(
            "Test 3 (+ Layer 2+3)",
            use_clahe=True,
            layers_config="deep",
            pool_size=3,
            blur_size=3,
        ),
        StrictAblationDetector(
            "Test 4 (Ours Full)",
            use_clahe=True,
            layers_config="deep",
            pool_size=5,
            blur_size=9,
        ),
    ]

    results = {}
    roc_data = {}

    for model in models_to_test:
        print(f"\n{'='*50}\n🚀 评估消融配置: {model.name}\n{'='*50}")
        model.fit(train_imgs, save_dir=LIB_SAVE_DIR)

        safe_name = (
            model.name.replace(" ", "_")
            .replace("+", "_")
            .replace("/", "_")
            .replace("&", "_")
        )
        scores_path = os.path.join(RESULTS_DIR, f"{safe_name}_scores.npy")

        # 依然保留结果缓存，不影响精度，只为了加速重复调试
        if os.path.exists(scores_path):
            print(f"⏭️ [{model.name}] 已检测到推理分数缓存，瞬间加载！")
            y_scores = np.load(scores_path)
        else:
            y_scores = []
            for img in tqdm(test_imgs, desc=f"[{model.name}] 正在推理"):
                y_scores.append(model.predict(img))
            y_scores = np.array(y_scores)
            np.save(scores_path, y_scores)

        y_true = np.array(test_labels)

        # 严格按照 Evaluate_metrics 结尾的最佳 F1 阈值逻辑计算
        auroc = roc_auc_score(y_true, y_scores)
        fpr_curve, tpr_curve, _ = roc_curve(y_true, y_scores)
        roc_data[model.name] = (fpr_curve, tpr_curve, auroc)

        max_good_score = np.max(y_scores[y_true == 0])

        precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
        f1_scores = np.divide(
            2 * precisions * recalls,
            precisions + recalls,
            out=np.zeros_like(precisions),
            where=(precisions + recalls) != 0,
        )

        best_idx = np.argmax(f1_scores)
        best_thresh = (
            thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]
        )

        y_preds_best = (y_scores >= best_thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_preds_best, labels=[0, 1]).ravel()
        best_fpr = fp / (fp + tn + 1e-8)

        results[model.name] = {
            "AUROC (%)": auroc * 100,
            "MaxScore": max_good_score,
            "FPR (%)": best_fpr * 100,
        }

    plot_ablation_results(results, roc_data)


def plot_ablation_results(results, roc_data):
    names = list(results.keys())
    aurocs = [results[n]["AUROC (%)"] for n in names]
    max_scores = [results[n]["MaxScore"] for n in names]
    fprs = [results[n]["FPR (%)"] for n in names]

    colors = ["#7f7f7f", "#1f77b4", "#ff7f0e", "#d62728"]

    plt.figure(figsize=(9, 7))
    for i, name in enumerate(names):
        fpr_curve, tpr_curve, auroc = roc_data[name]
        linewidth = 3 if "Ours" in name else 2
        plt.plot(
            fpr_curve,
            tpr_curve,
            color=colors[i],
            lw=linewidth,
            label=f"{name} (AUC = {auroc:.4f})",
        )

    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("Ablation Study: ROC Curves Comparison")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(RESULTS_DIR, "Ablation_Combined_ROC.png"), dpi=300)

    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, fprs, color=colors)
    plt.ylabel("False Positive Rate - FPR (%)")
    plt.title("Ablation Study: FPR Comparison (Lower is Better)")
    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 0.2,
            f"{yval:.2f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "Ablation_FPR_Comparison.png"), dpi=300)

    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, max_scores, color=colors)
    plt.ylabel("Max Anomaly Score on Good Samples")
    plt.title("Ablation Study: Background Noise Suppression (Lower is Better)")
    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 0.005,
            f"{yval:.4f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "Ablation_MaxScore_Comparison.png"), dpi=300)

    print("\n" + "=" * 80)
    print(
        f"{'Ablation Config':<30} | {'AUROC (%)':<10} | {'MaxScore':<10} | {'FPR (%)':<10}"
    )
    print("-" * 80)
    for n in names:
        print(
            f"{n:<30} | {results[n]['AUROC (%)']:<10.2f} | {results[n]['MaxScore']:<10.4f} | {results[n]['FPR (%)']:<10.2f}"
        )
    print("=" * 80)
    print(f"✅ 严格对齐完毕！图表已保存至 {RESULTS_DIR} 目录。")


if __name__ == "__main__":
    run_ablation_experiment()
