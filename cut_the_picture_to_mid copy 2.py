import cv2
import numpy as np
import os


def normalize_path(path):
    """处理Windows路径的反斜杠/中文问题"""
    return path.replace("\\", "/")


def segment_foreground(image_path):
    """分割主体（白织带+黑文字）与背景（黑边），适配当前图像特征"""
    # 读取图像（兼容中文/特殊路径）
    img = cv2.imdecode(
        np.fromfile(normalize_path(image_path), dtype=np.uint8), cv2.IMREAD_COLOR
    )
    if img is None:
        raise ValueError(f"图像读取失败，请检查路径：{image_path}")

    # 针对“黑边背景+白织带主体”的分割：用GrabCut提取前景（主体）
    mask = np.zeros(img.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    # 初始分割框：覆盖图像有效区域（避开边缘黑边）
    rect = (5, 5, img.shape[1] - 5, img.shape[0] - 5)
    cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

    # 掩码处理：将“确定前景+可能前景”标记为主体（255），背景为0
    mask = np.where((mask == 1) | (mask == 3), 255, 0).astype(np.uint8)

    # 膨胀操作：确保白织带和黑文字都被包含在主体中
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask, img


def get_foreground_bounds(mask):
    """提取主体（白织带+黑文字）的真实边界（去掉原黑边）"""
    coords = np.nonzero(mask)
    if len(coords[0]) == 0:
        raise ValueError("未检测到主体，请检查图像")

    min_row = np.min(coords[0])  # 主体最高点（原黑边的上边界）
    max_row = np.max(coords[0])  # 主体最低点（原黑边的下边界）
    min_col = np.min(coords[1])  # 主体左边缘
    max_col = np.max(coords[1])  # 主体右边缘
    return min_row, max_row, min_col, max_col


def pad_black_centered(
    img, mask, min_row, max_row, min_col, max_col, target_height=None
):
    """
    1. 先去掉原图多余黑边，提取纯净主体
    2. 以主体为中心，上下填充等量黑色像素（消除原黑边，生成新的居中矩形）
    """
    # 步骤1：提取纯净主体（去掉原黑边）
    pure_foreground = img[min_row : max_row + 1, min_col : max_col + 1]
    h, w = pure_foreground.shape[:2]  # 纯净主体的真实高度/宽度

    # 步骤2：确定最终图像高度（若未指定，默认主体高度的2倍，保证上下有足够黑边）
    final_h = target_height if target_height else h * 2
    final_w = w  # 宽度与主体一致（也可自定义）

    # 步骤3：计算上下填充量（保证上下黑色面积相等）
    pad_top = (final_h - h) // 2
    pad_bottom = final_h - h - pad_top  # 补全剩余像素（避免奇数差）

    # 步骤4：创建新画布，主体居中，上下填充黑色
    padded_img = np.zeros((final_h, final_w, 3), dtype=np.uint8)
    padded_img[pad_top : pad_top + h, :] = pure_foreground  # 主体放在中间

    return padded_img


def main(image_path, output_path="output_centered.png", target_height=None):
    try:
        # 1. 分割主体（去掉原黑边）
        mask, img = segment_foreground(image_path)

        # 2. 定位主体真实边界（纯净区域）
        min_row, max_row, min_col, max_col = get_foreground_bounds(mask)
        print(
            f"原黑边已去除，主体真实尺寸：宽={max_col-min_col+1}，高={max_row-min_row+1}"
        )

        # 3. 主体居中，上下填充等量黑边
        padded_img = pad_black_centered(
            img, mask, min_row, max_row, min_col, max_col, target_height=target_height
        )

        # 4. 保存结果
        cv2.imencode(os.path.splitext(output_path)[1], padded_img)[1].tofile(
            output_path
        )
        print(f"✅ 处理完成！结果保存至：{output_path}")
        print(f"📌 最终图像尺寸：宽={padded_img.shape[1]}，高={padded_img.shape[0]}")

    except Exception as e:
        print(f"❌ 处理失败：{str(e)}")


if __name__ == "__main__":
    # -------------------------- 你的路径配置 --------------------------
    image_path = r"F:\All_dataset\badest\\000000_251205_055.png"
    output_path = r"F:\All_dataset\badest"

    # 可选：指定最终图像高度（比如希望最终图高为200像素），None则自动为主体高度的2倍
    target_height = None

    # -------------------------- 执行处理 --------------------------
    main(image_path=image_path, output_path=output_path, target_height=target_height)
