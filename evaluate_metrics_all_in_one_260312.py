import os

# 【极速优化 1】限制底层数学库的线程，防止多线程打架导致 CPU 100% 卡顿
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"

import torch
import torch.nn as nn
import cv2
import numpy as np
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

# ===================== 1. 全局配置 =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_DIR = r"F:\All_dataset\\long_buliao_datasets_260107_cutbackground"
TEST_GOOD_DIR = r"F:\All_dataset\\260309buliao_datasets\\good"
TEST_BAD_DIR = r"F:\All_dataset\\260309buliao_datasets\bad"

# 结果保存目录
RESULTS_DIR = "Ablation_Results_260312"
os.makedirs(RESULTS_DIR, exist_ok=True)
LIB_SAVE_DIR = os.path.join(RESULTS_DIR, "feature_libs")
os.makedirs(LIB_SAVE_DIR, exist_ok=True)

INPUT_SIZE = (384, 1024)


# ===================== 2. 预处理函数 =====================
def robust_preprocess(img_path, use_clahe):
    try:
        img_original = cv2.imread(img_path)
        if img_original is None:
            return None
        target_h, target_w = INPUT_SIZE
        img_resized = cv2.resize(
            img_original, (target_w, target_h), interpolation=cv2.INTER_LINEAR
        )

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


# ===================== 3. 模型定义区 =====================
class AblationPatchCore:
    def __init__(
        self, name, use_clahe, target_layers, pool_size, blur_size, coreset_ratio=0.20
    ):
        self.name = name
        self.use_clahe = use_clahe
        self.target_layers = target_layers
        self.pool_size = pool_size
        self.blur_size = blur_size
        self.coreset_ratio = coreset_ratio

        base_model = resnet18(weights=ResNet18_Weights.DEFAULT)
        return_nodes = {layer: f"s{i+1}" for i, layer in enumerate(target_layers)}
        self.extractor = create_feature_extractor(
            base_model, return_nodes=return_nodes
        ).to(DEVICE)
        self.extractor.eval()

        padding = pool_size // 2
        self.avg_pool = nn.AvgPool2d(pool_size, stride=1, padding=padding)
        self.index = None

        # 记录 FAISS 状态
        self.is_gpu_faiss = False

    def fit(self, train_img_paths, save_dir):
        # 【修复路径报错】：将 "/" 和 "&" 替换掉，防止被系统误认为路径分隔符
        safe_name = (
            self.name.replace(" ", "_")
            .replace("+", "_")
            .replace("/", "_")
            .replace("&", "_")
        )
        index_path = os.path.join(save_dir, f"{safe_name}.ivf_index")

        # 1. 加载已有特征库
        if os.path.exists(index_path):
            print(f"🔄 [{self.name}] 加载已有特征库: {index_path}")
            self.index = faiss.read_index(index_path)
            self.index.nprobe = 5

            # 【修复 GPU 加速加载】
            if DEVICE.type == "cuda" and hasattr(faiss, "StandardGpuResources"):
                try:
                    res = faiss.StandardGpuResources()
                    self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                    self.is_gpu_faiss = True
                except Exception:
                    pass
            print(
                f"⚡ FAISS 运行模式: {'GPU (极速)' if self.is_gpu_faiss else 'CPU (受限)'}"
            )
            return

        # 2. 从头建库
        features_list = []
        for p in tqdm(train_img_paths, desc=f"[{self.name}] 正在建库"):
            tensor = robust_preprocess(p, self.use_clahe)
            if tensor is None:
                continue

            with torch.no_grad():
                outputs = self.extractor(tensor.to(DEVICE))
                keys = list(outputs.keys())
                base_feat = outputs[keys[0]]
                concat_feats = [base_feat]
                for k in keys[1:]:
                    up_feat = torch.nn.functional.interpolate(
                        outputs[k], size=base_feat.shape[-2:], mode="bilinear"
                    )
                    concat_feats.append(up_feat)

                features = torch.cat(concat_feats, dim=1)
                features = self.avg_pool(features)

                # 【极速优化 2】把极其耗时的 Norm 归一化操作留在 GPU 运算！
                B, C, H, W = features.shape
                features = features.view(C, H * W).t()  # 变形为 (H*W, C)
                # 使用 PyTorch 在 GPU 上算范数，比 CPU 的 numpy 快成百上千倍
                features = features / (
                    torch.norm(features, p=2, dim=1, keepdim=True) + 1e-8
                )
                f_np = features.cpu().numpy().astype(np.float32)

                n = f_np.shape[0]
                idx = np.random.choice(n, int(n * self.coreset_ratio), replace=False)
                features_list.append(f_np[idx])

        all_features = np.concatenate(features_list, axis=0)

        d = all_features.shape[1]
        nlist = max(32, min(int(8 * math.sqrt(all_features.shape[0])), 2048))
        quantizer = faiss.IndexFlatL2(d)
        self.index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
        self.index.train(all_features)
        self.index.add(all_features)
        self.index.nprobe = 5

        faiss.write_index(self.index, index_path)

        if DEVICE.type == "cuda" and hasattr(faiss, "StandardGpuResources"):
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                self.is_gpu_faiss = True
            except Exception:
                pass
        print(
            f"⚡ FAISS 运行模式: {'GPU (极速)' if self.is_gpu_faiss else 'CPU (受限)'}"
        )

    def predict(self, img_path):
        tensor = robust_preprocess(img_path, self.use_clahe)
        if tensor is None:
            return 0.0

        with torch.no_grad():
            outputs = self.extractor(tensor.to(DEVICE))
            keys = list(outputs.keys())
            base_feat = outputs[keys[0]]
            concat_feats = [base_feat]
            for k in keys[1:]:
                up_feat = torch.nn.functional.interpolate(
                    outputs[k], size=base_feat.shape[-2:], mode="bilinear"
                )
                concat_feats.append(up_feat)

            features = torch.cat(concat_feats, dim=1)
            features = self.avg_pool(features)

            # 【极速优化 3】推理时同样将 24576 个点的归一化留在 GPU 上处理！
            B, C, H, W = features.shape
            features = features.view(C, H * W).t()  # (H*W, C)
            features = features / (
                torch.norm(features, p=2, dim=1, keepdim=True) + 1e-8
            )

            # 转出 float32 直接给 FAISS 食用，零拷贝开销
            feat = features.cpu().numpy().astype(np.float32)

        distances, _ = self.index.search(feat, 9)

        score_map = np.mean(distances, axis=1).reshape(H, W)
        score_map = cv2.GaussianBlur(score_map, (self.blur_size, self.blur_size), 0)
        return np.max(score_map)


# ===================== 4. 消融实验流程 =====================
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

    models_to_test = [
        AblationPatchCore(
            "Test 1 (Baseline)",
            use_clahe=False,
            target_layers=["layer1", "layer2"],
            pool_size=3,
            blur_size=3,
        ),
        AblationPatchCore(
            "Test 2 (+ CLAHE)",
            use_clahe=True,
            target_layers=["layer1", "layer2"],
            pool_size=3,
            blur_size=3,
        ),
        AblationPatchCore(
            "Test 3 (+ Layer 2+3)",
            use_clahe=True,
            target_layers=["layer2", "layer3"],
            pool_size=3,
            blur_size=3,
        ),
        AblationPatchCore(
            "Test 4 (Ours / + K=5&Blur=9)",
            use_clahe=True,
            target_layers=["layer2", "layer3"],
            pool_size=5,
            blur_size=9,
        ),
    ]

    results = {}
    roc_data = {}

    for model in models_to_test:
        print(f"\n{'='*50}\n🚀 评估消融配置: {model.name}\n{'='*50}")
        # 1. 自动加载或构建特征库
        model.fit(train_imgs, save_dir=LIB_SAVE_DIR)

        # 定义分数缓存的路径 (同样清理非法字符)
        safe_name = (
            model.name.replace(" ", "_")
            .replace("+", "_")
            .replace("/", "_")
            .replace("&", "_")
        )
        scores_path = os.path.join(RESULTS_DIR, f"{safe_name}_scores.npy")

        # 2. 【新增】推理分数缓存逻辑
        if os.path.exists(scores_path):
            print(f"⏭️ [{model.name}] 检测到已有推理结果，直接加载分数，跳过推理环节！")
            y_scores = np.load(scores_path)
        else:
            y_scores = []
            start_time = time.time()
            for img in tqdm(test_imgs, desc=f"[{model.name}] 正在推理"):
                y_scores.append(model.predict(img))
            total_time = time.time() - start_time
            print(
                f"⏱️ {model.name} 推理耗时: {total_time:.2f} 秒 (FPS: {len(test_imgs)/total_time:.2f})"
            )
            y_scores = np.array(y_scores)
            # 将计算好的分数保存到硬盘，下次直接秒读
            np.save(scores_path, y_scores)

        y_true = np.array(test_labels)

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

        y_preds = (y_scores >= best_thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_preds, labels=[0, 1]).ravel()
        fpr = fp / (fp + tn + 1e-8)

        results[model.name] = {
            "AUROC (%)": auroc * 100,
            "MaxScore": max_good_score,
            "FPR (%)": fpr * 100,
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
            yval + 0.01,
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
    print(f"✅ 极速修复完成！图表已保存至 {RESULTS_DIR} 目录。")


if __name__ == "__main__":
    run_ablation_experiment()
