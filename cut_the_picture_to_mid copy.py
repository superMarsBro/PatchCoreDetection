import cv2
import numpy as np
from PIL import Image
import os


# 解决路径中反斜杠的转义问题
def normalize_path(path):
    return path.replace("\\", "/")


def segment_foreground(image_path, is_black_bg=True):
    """
    图像前景（主体）分割（优先纯黑背景快速分割，兼容复杂背景）
    :param image_path: 图像路径
    :param is_black_bg: 是否为纯黑背景（True=快速分割，False=GrabCut）
    :return: mask（主体掩码，255为主体，0为背景）、img（原始BGR图像）
    """
    # 读取图像（解决中文路径/转义问题）
    img = cv2.imdecode(
        np.fromfile(normalize_path(image_path), dtype=np.uint8), cv2.IMREAD_UNCHANGED
    )
    if img is None:
        raise ValueError(f"图像读取失败，请检查路径：{image_path}")

    # 纯黑背景快速分割（更高效，贴合“分离黑色背景”需求）
    if is_black_bg:
        # 转为灰度图，阈值分割（黑色背景=0，主体=255）
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)  # 阈值10区分纯黑背景
        return mask, img

    # 复杂背景：保留原GrabCut分割逻辑
    mask = np.zeros(img.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    rect = (1, 1, img.shape[1] - 2, img.shape[0] - 2)
    cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    mask = np.where((mask == 2) | (mask == 0), 0, 255).astype(np.uint8)
    return mask, img


def get_foreground_bounds(mask):
    """
    从掩码中提取主体的真实上下左右边界（最高点/最低点/左右边缘）
    :param mask: 主体掩码（255为主体，0为背景）
    :return: min_row(主体最高点), max_row(主体最低点), min_col(主体左边缘), max_col(主体右边缘)
    """
    coords = np.nonzero(mask)
    if len(coords[0]) == 0:
        raise ValueError("未检测到主体，请检查图像或分割参数")
    min_row = np.min(coords[0])  # 主体最高点（垂直方向最上方）
    max_row = np.max(coords[0])  # 主体最低点（垂直方向最下方）
    min_col = np.min(coords[1])  # 主体左边缘
    max_col = np.max(coords[1])  # 主体右边缘
    return min_row, max_row, min_col, max_col


def pad_black_centered(
    img, mask, min_row, max_row, min_col, max_col, target_aspect_ratio=None
):
    """
    核心逻辑：以主体真实边界为基准，填充黑色使主体居中，上下黑色面积相等
    :param img: 原始BGR图像
    :param mask: 主体掩码
    :param min_row/max_row: 主体最高点/最低点（垂直边界）
    :param min_col/max_col: 主体左右边缘（水平边界）
    :param target_aspect_ratio: 目标图像宽高比（如1:1=正方形，None=自适应主体宽高）
    :return: 填充后的BGR图像
    """
    # 1. 提取主体原始区域（仅保留主体像素，背景剔除）
    foreground = img[min_row : max_row + 1, min_col : max_col + 1]
    h, w = foreground.shape[:2]  # 主体自身高度/宽度（基于真实边界）

    # 2. 确定最终图像尺寸（保证主体居中，上下黑色面积相等）
    if target_aspect_ratio is None:
        # 方案1：最终图像宽度=主体宽度，高度=主体高度（仅上下填充等量黑边，宽度无填充）
        final_w = w
        final_h = h  # 可自定义最终高度，比如final_h = h * 2（上下各填h/2）
    else:
        # 方案2：按目标宽高比生成矩形（如正方形）
        final_w = max(w, int(h * target_aspect_ratio))
        final_h = max(h, int(final_w / target_aspect_ratio))

    # 3. 计算垂直方向填充（上下黑色像素数严格相等）
    pad_top = (final_h - h) // 2  # 上方填充黑像素数
    pad_bottom = (
        final_h - h - pad_top
    )  # 下方填充黑像素数（保证上下总和=final_h-h，且尽可能相等）
    # 水平方向填充（可选，保证主体水平居中）
    pad_left = (final_w - w) // 2
    pad_right = final_w - w - pad_left

    # 4. 创建纯黑画布（最终图像尺寸）
    padded_img = np.zeros((final_h, final_w, 3), dtype=np.uint8)

    # 5. 将主体放置在画布正中央（上下/左右填充等量黑色）
    padded_img[pad_top : pad_top + h, pad_left : pad_left + w] = foreground

    # 验证：上下黑色面积相等（像素数）
    top_black_pixels = pad_top * final_w
    bottom_black_pixels = pad_bottom * final_w
    if abs(top_black_pixels - bottom_black_pixels) > final_w:
        print(
            f"提示：上下黑色像素数差{abs(top_black_pixels - bottom_black_pixels)}（因高度为奇数）"
        )

    return padded_img


def main(
    image_path, output_path="output.png", target_aspect_ratio=None, is_black_bg=True
):
    """
    主函数：一键执行分割→找边界→填充黑色
    :param image_path: 输入图像路径
    :param output_path: 输出图像路径
    :param target_aspect_ratio: 目标宽高比（如1.0=正方形，2.0=宽是高的2倍）
    :param is_black_bg: 是否为纯黑背景（加速分割）
    """
    try:
        # 1. 分割主体（分离黑色背景）
        mask, img = segment_foreground(image_path, is_black_bg=is_black_bg)

        # 2. 获取主体真实边界（最高点/最低点/左右边缘）
        min_row, max_row, min_col, max_col = get_foreground_bounds(mask)

        # 3. 居中填充黑色（核心步骤）
        padded_img = pad_black_centered(
            img,
            mask,
            min_row,
            max_row,
            min_col,
            max_col,
            target_aspect_ratio=target_aspect_ratio,
        )

        # 4. 保存结果（解决中文路径保存问题）
        cv2.imencode(os.path.splitext(output_path)[1], padded_img)[1].tofile(
            output_path
        )
        print(f"✅ 处理完成！结果保存至：{output_path}")
        print(
            f"📌 主体边界：最高点={min_row}, 最低点={max_row}, 左边缘={min_col}, 右边缘={max_col}"
        )
        print(f"📌 最终图像尺寸：{padded_img.shape[1]}x{padded_img.shape[0]}（宽x高）")

        # 可选：显示结果（需GUI环境）
        # cv2.imshow("主体掩码", mask)
        # cv2.imshow("填充后图像", padded_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    except Exception as e:
        print(f"❌ 处理失败：{str(e)}")


if __name__ == "__main__":
    # -------------------------- 配置参数 --------------------------
    # 替换为你的图像路径（支持中文/反斜杠路径）
    image_path = r"F:\All_dataset\badest\\000000_251205_055.png"
    output_path = r"F:\All_dataset\badest\\output_centered.png"

    # 目标宽高比（按需调整）：
    # - None：仅上下填充，宽度=主体宽度
    # - 1.0：生成正方形（主体居中，上下/左右均填充等量黑色）
    # - 1.5：宽是高的1.5倍（矩形）
    target_aspect_ratio = None  # 优先贴合你的“主体边界为基准”需求

    # 是否为纯黑背景（True=快速分割，False=GrabCut复杂背景分割）
    is_black_bg = True

    # -------------------------- 执行处理 --------------------------
    main(
        image_path=image_path,
        output_path=output_path,
        target_aspect_ratio=target_aspect_ratio,
        is_black_bg=is_black_bg,
    )
