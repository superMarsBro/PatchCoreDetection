# 配置正常布料目录
NORMAL_DIR = "data/normal"  # 替换为你的正常布料目录
# 构建特征库
knn, feature_lib = build_feature_library(NORMAL_DIR)


# 单张检测示例
TEST_IMG_PATH = "data/test/defect_cloth.jpg"  # 替换为待检测图像
result_img, defect_mask, boxes = detect_defect(TEST_IMG_PATH, knn)

# 批量检测示例
def batch_detect(test_dir, knn):
    test_paths = []
    for root, _, files in os.walk(test_dir):
        for file in files:
            if file.lower().endswith(("jpg", "jpeg", "png")):
                test_paths.append(os.path.join(root, file))
    
    for img_path in test_paths:
        try:
            detect_defect(img_path, knn)
        except Exception as e:
            print(f"❌ 检测失败 {img_path}：{str(e)}")

# 运行批量检测
# batch_detect(test_dir="data/test", knn=knn)


#  后处理优化
# 单张检测示例
TEST_IMG_PATH = "data/test/defect_cloth.jpg"  # 替换为待检测图像
result_img, defect_mask, boxes = detect_defect(TEST_IMG_PATH, knn)

# 批量检测示例
def batch_detect(test_dir, knn):
    test_paths = []
    for root, _, files in os.walk(test_dir):
        for file in files:
            if file.lower().endswith(("jpg", "jpeg", "png")):
                test_paths.append(os.path.join(root, file))
    
    for img_path in test_paths:
        try:
            detect_defect(img_path, knn)
        except Exception as e:
            print(f"❌ 检测失败 {img_path}：{str(e)}")

# 运行批量检测
# batch_detect(test_dir="data/test", knn=knn)



# 光照鲁棒性优化
# 在normalize_light函数中补充
def normalize_light(img):
    # 新增：全局直方图均衡化（备用）
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if np.std(gray) < 30:  # 低对比度图像
        gray_eq = cv2.equalizeHist(gray)
        img = cv2.cvtColor(gray_eq, cv2.COLOR_GRAY2BGR)
    # 原有LAB处理
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    # ... 后续逻辑不变


# ONNX 模型导出（特征提取器）
def export_onnx(extractor, onnx_path="feature_extractor.onnx"):
    """导出特征提取器为ONNX格式"""
    dummy_input = torch.randn(1, 3, 512, 512).to(DEVICE)
    torch.onnx.export(
        extractor,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["features"],
        dynamic_axes={"input": {0: "batch_size"}, "features": {0: "batch_size"}},
        opset_version=12
    )
    print(f"✅ ONNX模型导出完成：{onnx_path}")

# 导出模型
# export_onnx(extractor)


# OpenVINO 加速推理
from openvino.runtime import Core

class OpenVINOExtractor:
    """OpenVINO加速的特征提取器"""
    def __init__(self, onnx_path):
        core = Core()
        model = core.read_model(onnx_path)
        self.compiled_model = core.compile_model(model, "CPU")
        self.output_layer = self.compiled_model.outputs[0]

    def __call__(self, x):
        # x: [1, 3, 512, 512] numpy数组
        result = self.compiled_model([x])[self.output_layer]
        return torch.from_numpy(result)

# 使用OpenVINO加速
# ov_extractor = OpenVINOExtractor("feature_extractor.onnx")