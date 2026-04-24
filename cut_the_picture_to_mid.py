import cv2
import numpy as np
import os
import glob
import argparse


def crop_and_pad_center(
    image_path,
    output_path=None,
    mode="square",  # "tight" / "square" / "height"
    target_height=None,  # mode="height" 时生效
    blur_ksize=5,
    morph_ksize=3,
    min_area=30,
    show_process=False,
):
    """
    1) 分离黑色背景(触边黑连通域)和主体(非触边黑连通域)
    2) 以主体bbox裁切(左右到主体边缘，上下到主体最高/最低)
    3) 只在上下填充黑色，使主体垂直居中，且上下黑边等高(等面积)
    4) 可选：square -> 输出高度=宽度（只上下补黑）
    """

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")

    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1) 去噪
    if blur_ksize and blur_ksize > 1:
        gray_blur = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    else:
        gray_blur = gray

    # 2) 阈值：把“黑色区域”变成1（暗->白）
    # 注意：这里用 INV
    _, dark = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 3) 形态学：让字母/主体更连贯，去小噪声
    if morph_ksize and morph_ksize > 1:
        k = morph_ksize
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel, iterations=1)
        dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 4) 连通域：触边的黑色连通域 -> 黑色背景；非触边 -> 主体
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        (dark > 0).astype(np.uint8), connectivity=8
    )

    def touches_border(x, y, w, h):
        return (x <= 0) or (y <= 0) or (x + w >= W) or (y + h >= H)

    black_bg_mask = np.zeros((H, W), dtype=np.uint8)
    subject_mask = np.zeros((H, W), dtype=np.uint8)

    for lab in range(1, num_labels):
        x, y, w, h, area = stats[lab]
        if area < min_area:
            continue
        comp = (labels == lab).astype(np.uint8) * 255
        if touches_border(x, y, w, h):
            black_bg_mask = cv2.bitwise_or(black_bg_mask, comp)
        else:
            subject_mask = cv2.bitwise_or(subject_mask, comp)

    # 极端情况：主体全部触边 -> 退化：取最大连通域当主体
    if cv2.countNonZero(subject_mask) == 0:
        best_lab, best_area = None, 0
        for lab in range(1, num_labels):
            x, y, w, h, area = stats[lab]
            if area > best_area:
                best_area = area
                best_lab = lab
        if best_lab is None:
            # 彻底没分出来
            if output_path:
                cv2.imwrite(output_path, img)
            return img
        subject_mask = (labels == best_lab).astype(np.uint8) * 255
        black_bg_mask = cv2.bitwise_and(dark, cv2.bitwise_not(subject_mask))

    # 5) 主体bbox（最高点/最低点/左右边缘）
    ys, xs = np.where(subject_mask > 0)
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()

    crop = img[y0 : y1 + 1, x0 : x1 + 1].copy()
    ch, cw = crop.shape[:2]

    # 6) 只上下补黑，保证上下等高，主体居中
    if mode == "tight":
        out_h = ch
    elif mode == "square":
        out_h = max(ch, cw)
    elif mode == "height":
        if target_height is None:
            raise ValueError("mode='height' 需要 target_height")
        out_h = max(target_height, ch)
    else:
        raise ValueError("mode 必须是 tight/square/height")

    pad_total = out_h - ch
    pad_top = pad_total // 2
    pad_bottom = pad_total - pad_top

    out = np.zeros((out_h, cw, 3), dtype=np.uint8)  # 黑底
    out[pad_top : pad_top + ch, :, :] = crop

    if output_path:
        cv2.imwrite(output_path, out)

    if show_process:
        # 仅用于调试：保存中间结果
        base = os.path.splitext(os.path.basename(image_path))[0]
        dbg_dir = os.path.join(os.path.dirname(output_path or "."), "_debug")
        os.makedirs(dbg_dir, exist_ok=True)
        cv2.imwrite(os.path.join(dbg_dir, f"{base}_dark.png"), dark)
        cv2.imwrite(os.path.join(dbg_dir, f"{base}_blackbg.png"), black_bg_mask)
        cv2.imwrite(os.path.join(dbg_dir, f"{base}_subject.png"), subject_mask)

    return out


def process_folder(input_folder, output_folder, overwrite=False, **kwargs):
    os.makedirs(output_folder, exist_ok=True)

    exts = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"]
    files = []
    for e in exts:
        files += glob.glob(os.path.join(input_folder, e))

    print(f"找到 {len(files)} 个图像文件")

    ok = skip = fail = 0
    for i, p in enumerate(files, 1):
        name = os.path.basename(p)
        outp = os.path.join(output_folder, name)

        if os.path.exists(outp) and not overwrite:
            skip += 1
            continue

        try:
            res = crop_and_pad_center(p, outp, **kwargs)
            ok += 1
            print(f"[{i}/{len(files)}] {name} -> {res.shape[1]}x{res.shape[0]}")
        except Exception as e:
            fail += 1
            print(f"[{i}/{len(files)}] 失败 {name}: {e}")

    print("\n完成：")
    print(f"成功: {ok}, 跳过: {skip}, 失败: {fail}")
    print(f"输出: {output_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="输入图像或文件夹")
    parser.add_argument("-o", "--output", required=True, help="输出图像或文件夹")
    parser.add_argument("--batch", action="store_true", help="批量处理")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在输出")
    parser.add_argument(
        "--mode", default="square", choices=["tight", "square", "height"]
    )
    parser.add_argument("--target-height", type=int, default=None)
    parser.add_argument("--show-process", action="store_true")
    args = parser.parse_args()

    if args.batch or os.path.isdir(args.input):
        process_folder(
            input_folder=args.input,
            output_folder=args.output,
            overwrite=args.overwrite,
            mode=args.mode,
            target_height=args.target_height,
            show_process=args.show_process,
        )
    else:
        crop_and_pad_center(
            args.input,
            args.output,
            mode=args.mode,
            target_height=args.target_height,
            show_process=args.show_process,
        )
        print(f"保存到: {args.output}")
