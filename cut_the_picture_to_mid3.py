import cv2
import numpy as np
from pathlib import Path


def segment_and_center_pad(
    img_path,
    out_dir="out",
    # 选择输出高度策略：
    # mode="tight"  : 紧贴主体bbox，只做最小padding(0) -> 仍满足上下相等(都是0)
    # mode="square" : 只在上下补黑边，让输出图尽量成为正方形(高度=宽度)
    # mode="height" : 指定 target_height，只在上下补黑边
    mode="square",
    target_height=None,
    # 阈值与形态学参数（一般默认就够用）
    blur_ksize=5,
    morph_ksize=3,
):
    img_path = str(img_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")

    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1) 轻微去噪
    if blur_ksize and blur_ksize > 1:
        gray_blur = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    else:
        gray_blur = gray

    # 2) Otsu 阈值：黑色区域(字母+黑条)会变成1
    # THRESH_BINARY_INV：暗->白(255)，亮->黑(0)
    _, dark = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 3) 形态学：去掉小噪点 + 让字母连贯些（可按需要调）
    k = morph_ksize
    if k and k > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel, iterations=1)
        dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 4) 连通域：把“触边”的黑色连通域当作黑色背景（上/下黑条等）
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        (dark > 0).astype(np.uint8), connectivity=8
    )

    def touches_border(x, y, w, h):
        return (x <= 0) or (y <= 0) or (x + w >= W) or (y + h >= H)

    black_bg_mask = np.zeros((H, W), dtype=np.uint8)
    subject_mask = np.zeros((H, W), dtype=np.uint8)

    for lab in range(1, num_labels):
        x, y, w, h, area = stats[lab]
        if area < 20:  # 小碎片直接丢掉
            continue
        comp = (labels == lab).astype(np.uint8) * 255
        if touches_border(x, y, w, h):
            black_bg_mask = cv2.bitwise_or(black_bg_mask, comp)
        else:
            subject_mask = cv2.bitwise_or(subject_mask, comp)

    # 如果主体被误分到触边里（极端情况），退化为：取“最大非空连通域”为主体
    if cv2.countNonZero(subject_mask) == 0:
        max_area = 0
        max_lab = None
        for lab in range(1, num_labels):
            x, y, w, h, area = stats[lab]
            if area > max_area:
                max_area = area
                max_lab = lab
        if max_lab is not None:
            subject_mask = (labels == max_lab).astype(np.uint8) * 255
            black_bg_mask = cv2.bitwise_and(dark, cv2.bitwise_not(subject_mask))

    # 5) 用主体最高点/最低点/左右边缘得到 bbox
    ys, xs = np.where(subject_mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        raise RuntimeError(
            "No subject found. Try adjusting morph_ksize/blur_ksize or threshold strategy."
        )

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()

    # 水平按主体边缘裁切；垂直先紧贴主体（后面只在上下补黑边）
    crop = img[y0 : y1 + 1, x0 : x1 + 1].copy()
    subj_crop = subject_mask[y0 : y1 + 1, x0 : x1 + 1].copy()
    bg_crop = black_bg_mask[y0 : y1 + 1, x0 : x1 + 1].copy()

    ch, cw = crop.shape[:2]

    # 6) 计算目标高度，并只在上下补黑边，保证上下等高（等面积）
    if mode == "tight":
        out_h = ch
    elif mode == "square":
        out_h = max(ch, cw)
    elif mode == "height":
        if target_height is None:
            raise ValueError("mode='height' requires target_height.")
        out_h = max(target_height, ch)
    else:
        raise ValueError("mode must be one of: tight/square/height")

    pad_total = out_h - ch
    pad_top = pad_total // 2
    pad_bottom = pad_total - pad_top  # 保证上下总和正确，且尽量相等

    # 7) 生成输出画布：上下填充黑色像素，使主体居中
    out = np.zeros((out_h, cw, 3), dtype=np.uint8)  # 黑底
    out[pad_top : pad_top + ch, :, :] = crop

    # 同步输出mask（可用于后续处理）
    out_subj = np.zeros((out_h, cw), dtype=np.uint8)
    out_subj[pad_top : pad_top + ch, :] = subj_crop

    out_blackbg = np.zeros((out_h, cw), dtype=np.uint8)
    out_blackbg[pad_top : pad_top + ch, :] = bg_crop

    # 8) 保存结果
    cv2.imwrite(str(out_dir / "01_dark_mask.png"), dark)
    cv2.imwrite(str(out_dir / "02_black_background_mask.png"), black_bg_mask)
    cv2.imwrite(str(out_dir / "03_subject_mask.png"), subject_mask)

    cv2.imwrite(str(out_dir / "04_subject_crop.png"), crop)
    cv2.imwrite(str(out_dir / "05_subject_center_padded.png"), out)

    cv2.imwrite(str(out_dir / "06_subject_mask_center_padded.png"), out_subj)
    cv2.imwrite(str(out_dir / "07_blackbg_mask_center_padded.png"), out_blackbg)

    print("Done.")
    print(f"Subject bbox in original: x[{x0},{x1}] y[{y0},{y1}]")
    print(f"Output size: {out.shape[1]}x{out.shape[0]} (W x H)")
    print(f"Pad top/bottom: {pad_top}/{pad_bottom}")
    return out, out_subj, out_blackbg


if __name__ == "__main__":
    # 示例：让输出尽量变成正方形（只在上下补黑边）
    segment_and_center_pad(
        img_path="F:\All_dataset\\badest\\000001_251205_000.png",
        out_dir="out",
        mode="square",
        morph_ksize=3,
        blur_ksize=5,
    )

    # 如果你想指定输出高度（比如固定到 200 像素高）：
    # segment_and_center_pad("xxx.png", out_dir="out", mode="height", target_height=200)
