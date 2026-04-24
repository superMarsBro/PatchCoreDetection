import cv2
import numpy as np
import os
import glob
import argparse


def crop_to_subject_bounds(
    image_path, output_path=None, show_process=False, method="otsu"
):
    """
    裁剪图像到主体边界，不添加额外背景

    参数:
        image_path: 输入图像路径
        output_path: 输出图像路径（可选）
        show_process: 是否显示处理过程
        method: 分割方法，可选 'otsu', 'adaptive', 'canny', 'all'

    返回:
        裁剪后的图像
    """
    # 1. 读取图像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")

    original_height, original_width = image.shape[:2]

    # 2. 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 3. 使用阈值分离背景
    if method == "otsu":
        # 使用OTSU阈值
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    elif method == "adaptive":
        # 使用自适应阈值
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

    elif method == "canny":
        # 使用边缘检测
        binary = cv2.Canny(gray, 30, 100)

    elif method == "all":
        # 组合多种方法
        _, binary_otsu = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        binary_adaptive = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        binary_canny = cv2.Canny(gray, 30, 100)

        # 组合结果
        binary = cv2.bitwise_or(binary_otsu, binary_adaptive)
        binary = cv2.bitwise_or(binary, binary_canny)
    else:
        raise ValueError(f"未知的分割方法: {method}")

    # 4. 形态学操作去除噪声
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # 5. 寻找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print(f"警告: 未找到主体轮廓，使用全图: {image_path}")
        if output_path:
            cv2.imwrite(output_path, image)
        return image

    # 找到最大的轮廓
    largest_contour = max(contours, key=cv2.contourArea)

    # 6. 获取主体的精确边界
    # 获取边界矩形
    x, y, w, h = cv2.boundingRect(largest_contour)

    # 确保边界在合理范围内
    x = max(0, x)
    y = max(0, y)
    w = min(w, original_width - x)
    h = min(h, original_height - y)

    # 如果边界框太小，可能是噪声，使用全图
    min_area_ratio = 0.01  # 最小面积比例
    if w * h < original_width * original_height * min_area_ratio:
        print(f"警告: 主体区域过小，使用全图: {image_path}")
        if output_path:
            cv2.imwrite(output_path, image)
        return image

    # 7. 裁剪图像
    cropped_image = image[y : y + h, x : x + w]

    # 8. 保存结果
    if output_path:
        cv2.imwrite(output_path, cropped_image)

    # 9. 显示处理过程（如果需要）
    if show_process:
        # 创建掩码
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [largest_contour], -1, 255, -1)

        # 提取主体
        subject = cv2.bitwise_and(image, image, mask=mask)

        # 显示结果
        cv2.imshow("1. Original Image", image)
        cv2.imshow("2. Binary Image", binary)
        cv2.imshow("3. Subject Mask", mask)
        cv2.imshow("4. Extracted Subject", subject)
        cv2.imshow("5. Cropped Result", cropped_image)

        print("=" * 50)
        print(f"图像: {os.path.basename(image_path)}")
        print(f"原始图像尺寸: {original_width}x{original_height}")
        print(f"主体边界框: x={x}, y={y}, w={w}, h={h}")
        print(f"裁剪后尺寸: {w}x{h}")
        print("=" * 50)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return cropped_image


def process_folder(
    input_folder, output_folder=None, method="otsu", show_progress=True, overwrite=False
):
    """
    处理整个文件夹内的图像

    参数:
        input_folder: 输入文件夹路径
        output_folder: 输出文件夹路径，None则创建在输入文件夹下的'cropped'子文件夹
        method: 分割方法
        show_progress: 是否显示处理进度
        overwrite: 是否覆盖已存在的输出文件
    """
    # 检查输入文件夹是否存在
    if not os.path.exists(input_folder):
        raise ValueError(f"输入文件夹不存在: {input_folder}")

    # 设置输出文件夹
    if output_folder is None:
        output_folder = os.path.join(input_folder, "cropped")

    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"创建输出文件夹: {output_folder}")

    # 支持的图像格式
    image_extensions = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff", "*.tif"]
    image_files = []

    # 收集所有图像文件
    for ext in image_extensions:
        pattern = os.path.join(input_folder, ext)
        image_files.extend(glob.glob(pattern))

    if not image_files:
        print(f"在文件夹中未找到图像文件: {input_folder}")
        return

    print(f"找到 {len(image_files)} 个图像文件")

    # 处理每个图像
    processed_count = 0
    skipped_count = 0
    failed_count = 0

    for i, input_path in enumerate(image_files):
        try:
            # 生成输出路径
            filename = os.path.basename(input_path)
            output_path = os.path.join(output_folder, filename)

            # 检查是否已存在且不覆盖
            if os.path.exists(output_path) and not overwrite:
                if show_progress:
                    print(f"跳过已存在的文件: {filename}")
                skipped_count += 1
                continue

            # 处理图像
            if show_progress:
                print(f"处理 [{i+1}/{len(image_files)}]: {filename}")

            result = crop_to_subject_bounds(
                input_path,
                output_path,
                show_process=False,  # 批量处理时不显示过程
                method=method,
            )

            if show_progress:
                print(
                    f"  原始: {os.path.getsize(input_path) // 1024} KB, "
                    f"裁剪后: {result.shape[1]}x{result.shape[0]}"
                )

            processed_count += 1

        except Exception as e:
            print(f"处理失败 {os.path.basename(input_path)}: {str(e)}")
            failed_count += 1

    # 打印总结
    print("\n" + "=" * 50)
    print("处理完成!")
    print(f"总共: {len(image_files)} 个文件")
    print(f"成功处理: {processed_count} 个")
    print(f"跳过: {skipped_count} 个")
    print(f"失败: {failed_count} 个")
    print(f"输出文件夹: {output_folder}")
    print("=" * 50)


def main_single_image():
    """处理单个图像"""
    input_image = r"F:\All_dataset\badest\000000_251205_055.png"
    output_image = "cropped_result.png"

    print("处理单个图像...")
    result = crop_to_subject_bounds(
        input_image, output_image, show_process=True, method="all"
    )

    # 显示最终结果
    original = cv2.imread(input_image)
    cv2.imshow("Original", original)
    cv2.imshow("Cropped Result", result)

    print("\n最终结果:")
    print(f"原始图像: {original.shape[1]}x{original.shape[0]}")
    print(f"裁剪后图像: {result.shape[1]}x{result.shape[0]}")
    print(f"保存到: {output_image}")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main_batch():
    """批量处理文件夹"""
    input_folder = r"F:\All_dataset\buliao_dataset251208"
    output_folder = r"F:\All_dataset\buliao_cropped251223"

    print("开始批量处理...")
    print(f"输入文件夹: {input_folder}")
    print(f"输出文件夹: {output_folder}")

    # 可选的分割方法: 'otsu', 'adaptive', 'canny', 'all'
    segmentation_method = "all"

    process_folder(
        input_folder=input_folder,
        output_folder=output_folder,
        method=segmentation_method,
        show_progress=True,
        overwrite=False,  # 设置为True可以覆盖已存在的文件
    )


if __name__ == "__main__":
    # 使用argparse处理命令行参数
    parser = argparse.ArgumentParser(description="裁剪图像到主体边界")
    parser.add_argument("--input", "-i", help="输入图像或文件夹路径")
    parser.add_argument("--output", "-o", help="输出图像或文件夹路径")
    parser.add_argument(
        "--method",
        "-m",
        default="all",
        choices=["otsu", "adaptive", "canny", "all"],
        help="分割方法 (默认: all)",
    )
    parser.add_argument("--batch", "-b", action="store_true", help="批量处理模式")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在的输出文件")
    parser.add_argument("--no-progress", action="store_true", help="不显示处理进度")

    args = parser.parse_args()

    # 如果指定了命令行参数，使用它们
    if args.input:
        if args.batch or os.path.isdir(args.input):
            # 批量处理模式
            process_folder(
                input_folder=args.input,
                output_folder=args.output,
                method=args.method,
                show_progress=not args.no_progress,
                overwrite=args.overwrite,
            )
        else:
            # 单图像处理模式
            result = crop_to_subject_bounds(
                args.input, args.output, show_process=True, method=args.method
            )

            # 显示结果
            if args.output:
                print(f"处理完成，结果保存到: {args.output}")

            cv2.imshow("Cropped Result", result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        # 如果没有命令行参数，使用硬编码的路径
        print("未指定输入路径，使用硬编码路径...")

        # 选择处理模式
        mode = "batch"  # 可选 "single" 或 "batch"

        if mode == "single":
            main_single_image()
        else:
            main_batch()
