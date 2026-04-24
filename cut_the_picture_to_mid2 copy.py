import cv2
import numpy as np
import os


def segment_and_crop_to_center(image_path, output_path=None, padding_ratio=0.0):
    """
    以主体最高点和最低点为上下边界，裁剪并居中主体

    参数:
        image_path: 输入图像路径
        output_path: 输出图像路径（可选）
        padding_ratio: 在主体周围添加额外黑色背景的比例，0表示无额外背景

    返回:
        处理后的图像
    """
    # 1. 读取图像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("无法读取图像，请检查文件路径")

    original_height, original_width = image.shape[:2]

    # 2. 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 3. 使用多种方法尝试分离背景
    # 方法1: 使用自适应阈值
    binary1 = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # 方法2: 使用OTSU阈值
    _, binary2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 方法3: 使用边缘检测
    edges = cv2.Canny(gray, 30, 100)

    # 组合多种方法的结果
    combined = cv2.bitwise_or(binary1, binary2)
    combined = cv2.bitwise_or(combined, edges)

    # 4. 形态学操作去除噪点和连接区域
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

    # 5. 寻找轮廓
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("未找到主体轮廓，使用全图")
        return image

    # 找到最大的轮廓（假设主体是最大的）
    largest_contour = max(contours, key=cv2.contourArea)

    # 6. 获取主体的精确边界
    # 找到主体的最小y（最高点）和最大y（最低点）
    y_coords = largest_contour[:, 0, 1]
    min_y = np.min(y_coords)
    max_y = np.max(y_coords)

    # 找到主体的最小x（最左点）和最大x（最右点）
    x_coords = largest_contour[:, 0, 0]
    min_x = np.min(x_coords)
    max_x = np.max(x_coords)

    # 确保边界在图像范围内
    min_y = max(0, min_y)
    max_y = min(original_height - 1, max_y)
    min_x = max(0, min_x)
    max_x = min(original_width - 1, max_x)

    # 计算主体高度和宽度
    subject_height = max_y - min_y + 1
    subject_width = max_x - min_x + 1

    # 7. 裁剪出主体
    subject = image[min_y : max_y + 1, min_x : max_x + 1]

    # 8. 创建新图像，主体居中
    # 计算新图像的尺寸
    # 高度: 主体高度
    # 宽度: 主体宽度 + 两侧的黑色区域
    new_height = subject_height
    new_width = subject_width

    # 添加额外的黑色背景（如果padding_ratio > 0）
    if padding_ratio > 0:
        padding_height = int(subject_height * padding_ratio)
        padding_width = int(subject_width * padding_ratio)
        new_height += 2 * padding_height
        new_width += 2 * padding_width

    # 创建黑色背景图像
    centered_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)

    # 计算放置位置
    if padding_ratio > 0:
        start_x = padding_width
        start_y = padding_height
    else:
        start_x = 0
        start_y = 0

    # 将主体放置到新图像中
    centered_image[
        start_y : start_y + subject_height, start_x : start_x + subject_width
    ] = subject

    # 9. 保存结果
    if output_path:
        cv2.imwrite(output_path, centered_image)
        print(f"处理后的图像已保存到: {output_path}")

    return centered_image


def crop_to_subject_bounds(image_path, output_path=None, show_process=False):
    """
    裁剪图像到主体边界，不添加额外背景

    参数:
        image_path: 输入图像路径
        output_path: 输出图像路径（可选）
        show_process: 是否显示处理过程

    返回:
        裁剪后的图像
    """
    # 1. 读取图像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("无法读取图像，请检查文件路径")

    original_height, original_width = image.shape[:2]

    # 2. 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 3. 使用阈值分离背景
    # 尝试多个阈值方法找到最佳分离效果
    methods = [
        ("OTSU", cv2.THRESH_BINARY + cv2.THRESH_OTSU),
        ("BINARY_INV", cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU),
    ]

    best_contour = None
    best_area = 0

    for method_name, method in methods:
        # 应用阈值
        _, binary = cv2.threshold(gray, 0, 255, method)

        # 形态学操作去除噪声
        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # 寻找轮廓
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:
            # 找到最大的轮廓
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)

            # 选择面积最大的有效轮廓
            if area > best_area and area > original_width * original_height * 0.01:
                best_area = area
                best_contour = largest_contour

    if best_contour is None:
        print("未找到有效主体轮廓，使用全图")
        if output_path:
            cv2.imwrite(output_path, image)
        return image

    # 4. 获取主体的精确边界
    # 获取边界矩形
    x, y, w, h = cv2.boundingRect(best_contour)

    # 确保边界在合理范围内
    x = max(0, x)
    y = max(0, y)
    w = min(w, original_width - x)
    h = min(h, original_height - y)

    # 5. 裁剪图像
    cropped_image = image[y : y + h, x : x + w]

    # 6. 保存结果
    if output_path:
        cv2.imwrite(output_path, cropped_image)
        print(f"裁剪后的图像已保存到: {output_path}")

    # 7. 显示处理过程（如果需要）
    if show_process:
        # 创建掩码
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [best_contour], -1, 255, -1)

        # 提取主体
        subject = cv2.bitwise_and(image, image, mask=mask)

        # 显示结果
        cv2.imshow("1. Original Image", image)
        cv2.imshow("2. Subject Mask", mask)
        cv2.imshow("3. Extracted Subject", subject)
        cv2.imshow("4. Cropped Result", cropped_image)

        print("=" * 50)
        print(f"原始图像尺寸: {original_width}x{original_height}")
        print(f"主体边界框: x={x}, y={y}, w={w}, h={h}")
        print(f"裁剪后尺寸: {w}x{h}")
        print("=" * 50)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return cropped_image


def process_dataset(input_dir, output_dir):
    """
    处理整个数据集

    参数:
        input_dir: 输入目录
        output_dir: 输出目录
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 支持的图像格式
    image_extensions = [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]

    # 遍历输入目录
    for filename in os.listdir(input_dir):
        # 检查文件扩展名
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            try:
                print(f"处理: {filename}")
                # 使用裁剪函数
                result = crop_to_subject_bounds(
                    input_path, output_path, show_process=False  # 批量处理时不显示过程
                )
                print(f"  完成: {result.shape[1]}x{result.shape[0]}")
            except Exception as e:
                print(f"  处理失败: {str(e)}")


def main():
    """
    主函数：处理单个图像
    """
    # 单个图像处理示例
    input_image = r"F:\All_dataset\badest\\000000_251205_055.png"
    output_image = "cropped_result.png"

    # 方法1：裁剪到主体边界
    print("方法1：裁剪到主体边界")
    result1 = crop_to_subject_bounds(input_image, output_image, show_process=True)

    # 方法2：裁剪并居中（可选添加背景）
    print("\n方法2：裁剪并居中（添加10%背景）")
    output_image2 = "centered_result.png"
    result2 = segment_and_crop_to_center(
        input_image, output_image2, padding_ratio=0.1  # 添加10%的背景
    )

    # 显示最终结果对比
    original = cv2.imread(input_image)
    cv2.imshow("Original", original)
    cv2.imshow("Cropped to Subject", result1)
    cv2.imshow("Centered with Background", result2)

    print("\n最终结果对比:")
    print(f"原始图像: {original.shape[1]}x{original.shape[0]}")
    print(f"裁剪后图像: {result1.shape[1]}x{result1.shape[0]}")
    print(f"居中后图像: {result2.shape[1]}x{result2.shape[0]}")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def batch_process():
    """
    批量处理函数
    """
    input_dir = r"F:\All_dataset\badest"
    output_dir = r"F:\All_dataset\badest_cropped"

    print(f"开始处理数据集...")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")

    process_dataset(input_dir, output_dir)

    print("批量处理完成！")


if __name__ == "__main__":
    # 选择处理模式
    mode = "single"  # 可选 "single" 或 "batch"

    if mode == "single":
        # 处理单个图像
        main()
    else:
        # 批量处理
        batch_process()
