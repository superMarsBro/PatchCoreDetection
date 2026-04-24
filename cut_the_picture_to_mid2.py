import cv2
import numpy as np
import os
import glob
import argparse


def detect_bottom_black_background(image, black_threshold=20, ratio_threshold=0.6):
    """
    检测图片底部是否有大面积黑色背景
    :param image: 输入图像（BGR）
    :param black_threshold: 黑色像素的灰度阈值（<30判定为黑）
    :param ratio_threshold: 底部区域黑色像素占比阈值（>50%判定为黑背景）
    :return: (是否有黑背景, 黑背景的上边界y坐标)
    """
    height, width = image.shape[:2]
    # 取图片底部1/5区域检测（可根据实际调整）
    bottom_region_height = max(20, height // 4)
    bottom_region = image[height - bottom_region_height : height, :]

    # 转灰度图，统计黑色像素占比
    gray_bottom = cv2.cvtColor(bottom_region, cv2.COLOR_BGR2GRAY)
    black_pixels = np.sum(gray_bottom < black_threshold)
    total_pixels = bottom_region_height * width
    black_ratio = black_pixels / total_pixels

    if black_ratio > ratio_threshold:
        # 向上找第一个非黑区域的边界（精准裁切黑背景）
        for y in range(height - bottom_region_height, 0, -1):
            row_gray = cv2.cvtColor(image[y : y + 1, :], cv2.COLOR_BGR2GRAY)
            row_black_ratio = np.sum(row_gray < black_threshold) / width
            if row_black_ratio < ratio_threshold:
                return True, y + 1  # 黑背景的上边界
        return True, 0
    return False, height


def crop_to_subject_bounds(
    image_path, output_path=None, show_process=False, method="otsu", crop_black_bg=True
):
    """
    裁剪图像到主体边界，仅对底部黑背景图片做二次裁切（不影响其他图片）
    :param crop_black_bg: 是否启用底部黑背景裁切（核心开关）
    """
    # 1. 读取图像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")

    original_height, original_width = image.shape[:2]

    # 2. 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 3. 阈值分割（恢复原代码参数，保证大部分图片正常）
    if method == "otsu":
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == "adaptive":
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2,  # 恢复原参数
        )
    elif method == "canny":
        binary = cv2.Canny(gray, 30, 100)
    elif method == "all":
        _, binary_otsu = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        binary_adaptive = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        binary_canny = cv2.Canny(gray, 30, 100)
        binary = cv2.bitwise_or(binary_otsu, binary_adaptive)
        binary = cv2.bitwise_or(binary, binary_canny)
    else:
        raise ValueError(f"未知的分割方法: {method}")

    # 4. 形态学操作（恢复原参数）
    kernel = np.ones((3, 3), np.uint8)  # 恢复3x3 kernel
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

    # 6. 获取主体边界（恢复原参数）
    x, y, w, h = cv2.boundingRect(largest_contour)
    x = max(0, x)
    y = max(0, y)
    w = min(w, original_width - x)
    h = min(h, original_height - y)

    # 最小面积比例恢复为0.01（避免小噪声被识别）
    min_area_ratio = 0.01
    if w * h < original_width * original_height * min_area_ratio:
        print(f"警告: 主体区域过小，使用全图: {image_path}")
        if output_path:
            cv2.imwrite(output_path, image)
        return image

    # 7. 基础裁剪（原逻辑，保证大部分图片正常）
    cropped_image = image[y : y + h, x : x + w]

    # 8. 针对性处理：仅对底部有黑背景的图片做二次裁切
    if crop_black_bg:
        has_black_bg, bg_upper_y = detect_bottom_black_background(cropped_image)
        if has_black_bg:
            # 二次裁切：去掉底部黑背景
            cropped_image = cropped_image[:bg_upper_y, :]

    # 9. 保存结果
    if output_path:
        cv2.imwrite(output_path, cropped_image)

    # 10. 显示处理过程（原逻辑）
    if show_process:
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [largest_contour], -1, 255, -1)
        subject = cv2.bitwise_and(image, image, mask=mask)
        cv2.imshow("1. Original Image", image)
        cv2.imshow("2. Binary Image", binary)
        cv2.imshow("3. Subject Mask", mask)
        cv2.imshow("4. Extracted Subject", subject)
        cv2.imshow("5. Cropped Result", cropped_image)
        print("=" * 50)
        print(f"图像: {os.path.basename(image_path)}")
        print(f"原始图像尺寸: {original_width}x{original_height}")
        print(f"主体边界框: x={x}, y={y}, w={w}, h={h}")
        print(f"裁剪后尺寸: {cropped_image.shape[1]}x{cropped_image.shape[0]}")
        print("=" * 50)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return cropped_image


def process_folder(
    input_folder,
    output_folder=None,
    method="otsu",
    show_progress=True,
    overwrite=False,
    crop_black_bg=True,
):
    """批量处理文件夹，新增crop_black_bg开关"""
    if not os.path.exists(input_folder):
        raise ValueError(f"输入文件夹不存在: {input_folder}")
    if output_folder is None:
        output_folder = os.path.join(input_folder, "cropped")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"创建输出文件夹: {output_folder}")

    image_extensions = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff", "*.tif"]
    image_files = []
    for ext in image_extensions:
        pattern = os.path.join(input_folder, ext)
        image_files.extend(glob.glob(pattern))

    if not image_files:
        print(f"在文件夹中未找到图像文件: {input_folder}")
        return

    print(f"找到 {len(image_files)} 个图像文件")
    processed_count = 0
    skipped_count = 0
    failed_count = 0

    for i, input_path in enumerate(image_files):
        try:
            filename = os.path.basename(input_path)
            output_path = os.path.join(output_folder, filename)
            if os.path.exists(output_path) and not overwrite:
                if show_progress:
                    print(f"跳过已存在的文件: {filename}")
                skipped_count += 1
                continue
            if show_progress:
                print(f"处理 [{i+1}/{len(image_files)}]: {filename}")
            result = crop_to_subject_bounds(
                input_path,
                output_path,
                show_process=False,
                method=method,
                crop_black_bg=crop_black_bg,  # 传递黑背景裁切开关
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
        input_image, output_image, show_process=True, method="all", crop_black_bg=True
    )
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
    input_folder = r"F:\All_dataset\cut_test"
    output_folder = r"F:\All_dataset\badest_cropped"
    print("开始批量处理...")
    print(f"输入文件夹: {input_folder}")
    print(f"输出文件夹: {output_folder}")
    segmentation_method = "all"
    process_folder(
        input_folder=input_folder,
        output_folder=output_folder,
        method=segmentation_method,
        show_progress=True,
        overwrite=False,
        crop_black_bg=True,  # 启用黑背景裁切
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="裁剪图像到主体边界（兼容黑背景裁切）")
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
    parser.add_argument(
        "--no-black-bg-crop",
        action="store_true",
        help="禁用底部黑背景裁切（恢复纯原逻辑）",
    )

    args = parser.parse_args()

    if args.input:
        crop_black_bg = not args.no_black_bg_crop  # 命令行控制是否裁切黑背景
        if args.batch or os.path.isdir(args.input):
            process_folder(
                input_folder=args.input,
                output_folder=args.output,
                method=args.method,
                show_progress=not args.no_progress,
                overwrite=args.overwrite,
                crop_black_bg=crop_black_bg,
            )
        else:
            result = crop_to_subject_bounds(
                args.input,
                args.output,
                show_process=True,
                method=args.method,
                crop_black_bg=crop_black_bg,
            )
            if args.output:
                print(f"处理完成，结果保存到: {args.output}")
            cv2.imshow("Cropped Result", result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        print("未指定输入路径，使用硬编码路径...")
        mode = "batch"  # 可选 "single" 或 "batch"
        if mode == "single":
            main_single_image()
        else:
            main_batch()
