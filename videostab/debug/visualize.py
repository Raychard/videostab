"""推理各阶段可视化 (纯 OpenCV, 零新依赖).

坐标系: 全部在代理分辨率 (proxy) 像素系; 运动/校正向量幅值通常仅几像素,
统一按 ARROW_SCALE 放大绘制并在图上标注比例.
"""
import cv2
import numpy as np

ARROW_SCALE = 4.0                       # 运动/校正箭头放大倍数
GREEN, RED, YELLOW = (0, 200, 0), (0, 0, 230), (0, 200, 230)
CYAN, ORANGE, WHITE = (230, 230, 0), (0, 140, 255), (255, 255, 255)
LEVEL_COLORS = {0: GREEN, 1: YELLOW, 2: RED}
LEVEL_NAMES = {0: "L0", 1: "L1", 2: "L2"}


def _bgr(gray: np.ndarray) -> np.ndarray:
    if gray.ndim == 2:
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return gray.copy()


def _label(img, text, org=(6, 18), color=WHITE, scale=0.5):
    """带半透明黑底的文字, 保证任意背景可读."""
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x, y = org
    cv2.rectangle(img, (x - 3, y - th - 3), (x + tw + 3, y + 4), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1,
                cv2.LINE_AA)


def _arrow(img, p, v, color, scale=ARROW_SCALE):
    p0 = (int(round(p[0])), int(round(p[1])))
    p1 = (int(round(p[0] + v[0] * scale)), int(round(p[1] + v[1] * scale)))
    cv2.arrowedLine(img, p0, p1, color, 1, cv2.LINE_AA, tipLength=0.3)


def draw_keypoints_flow(gray, kept_pts, kept_motions,
                        rej_pts=None, rej_motions=None, header="") -> np.ndarray:
    """Stage1: 背景关键点(绿)+光流箭头, 被剔除的前景/误跟踪点(红)."""
    img = _bgr(gray)
    if rej_pts is not None:
        for p, v in zip(rej_pts, rej_motions):
            cv2.circle(img, (int(p[0]), int(p[1])), 2, RED, -1)
            _arrow(img, p, v, RED)
    for p, v in zip(kept_pts, kept_motions):
        cv2.circle(img, (int(p[0]), int(p[1])), 2, GREEN, -1)
        _arrow(img, p, v, GREEN)
    n_rej = 0 if rej_pts is None else len(rej_pts)
    _label(img, f"Stage1 kp={len(kept_pts)} rej={n_rej} (x{ARROW_SCALE:g})")
    if header:
        _label(img, header, org=(6, img.shape[0] - 8), color=CYAN)
    return img


def draw_grid_motion(shape_hw, grid, K=None, grid_err=None,
                     level=None) -> np.ndarray:
    """Stage2: 网格顶点运动场箭头(按幅值着色), 标注平面数 K 与拟合残差."""
    h, w = shape_hw
    img = np.full((h, w, 3), 30, np.uint8)
    gh, gw = grid.shape[:2]
    xs = np.linspace(0, w - 1, gw)
    ys = np.linspace(0, h - 1, gh)
    mag = np.linalg.norm(grid, axis=-1)
    mmax = max(mag.max(), 1e-6)
    for j in range(gh):
        for i in range(gw):
            p = (xs[i], ys[j])
            v = grid[j, i]
            t = mag[j, i] / mmax
            color = (int(255 * (1 - t)), int(120 + 100 * t), int(255 * t))
            cv2.circle(img, (int(p[0]), int(p[1])), 1, (90, 90, 90), -1)
            _arrow(img, p, v, color)
    parts = ["Stage2 grid-motion"]
    if K is not None:
        parts.append(f"K={K}")
    if grid_err is not None:
        parts.append(f"err={grid_err:.2f}px")
    if level is not None:
        parts.append(LEVEL_NAMES.get(int(level), "?"))
    _label(img, "  ".join(parts))
    _label(img, f"max|m|={mmax:.2f}px (x{ARROW_SCALE:g})",
           org=(6, h - 8), color=CYAN)
    return img


def draw_correction_field(shape_hw, B_frame, crop_ratio) -> np.ndarray:
    """Stage4: 逐顶点校正向量 + 裁剪预算框(绿框内为最终保留区域)."""
    h, w = shape_hw
    img = np.full((h, w, 3), 30, np.uint8)
    my, mx = int(h * crop_ratio / 2), int(w * crop_ratio / 2)
    cv2.rectangle(img, (mx, my), (w - mx, h - my), GREEN, 1)
    gh, gw = B_frame.shape[:2]
    xs = np.linspace(0, w - 1, gw)
    ys = np.linspace(0, h - 1, gh)
    for j in range(gh):
        for i in range(gw):
            _arrow(img, (xs[i], ys[j]), B_frame[j, i], ORANGE)
    _label(img, f"Stage4 correction  crop={crop_ratio:.0%} "
                f"max|B|={np.abs(B_frame).max():.2f}px (x{ARROW_SCALE:g})")
    return img


def _line_chart(canvas, series, colors, labels, y0, y1, title):
    """在 canvas 指定竖直区间 [y0,y1) 画多条时间序列折线(共享 x=帧)."""
    h_reg = y1 - y0
    W = canvas.shape[1]
    lo = min(s.min() for s in series)
    hi = max(s.max() for s in series)
    rng = max(hi - lo, 1e-6)
    T = len(series[0])
    cv2.line(canvas, (0, y1 - 1), (W, y1 - 1), (70, 70, 70), 1)
    for s, c in zip(series, colors):
        pts = []
        for t in range(T):
            x = int(t / max(T - 1, 1) * (W - 1))
            y = int(y1 - 1 - (s[t] - lo) / rng * (h_reg - 14))
            pts.append((x, y))
        cv2.polylines(canvas, [np.array(pts, np.int32)], False, c, 1,
                      cv2.LINE_AA)
    _label(canvas, title, org=(6, y0 + 16))
    for k, (lab, c) in enumerate(zip(labels, colors)):
        _label(canvas, lab, org=(90 + k * 90, y0 + 16), color=c)
    _label(canvas, f"[{lo:.1f},{hi:.1f}]", org=(W - 92, y0 + 16), color=(160,)*3)


def plot_paths(C, P, B=None, size=(720, 480)) -> np.ndarray:
    """全局: 相机路径 C(原始, 抖动) vs P(平滑) 的 x/y 均值曲线对比.

    C/P: (T,GH,GW,2). 曲线越接近说明平滑越弱; 差值即校正量 B.
    """
    W, H = size
    canvas = np.full((H, W, 3), 24, np.uint8)
    cm = C.mean(axis=(1, 2))          # (T,2)
    pm = P.mean(axis=(1, 2))
    half = H // 2
    _line_chart(canvas, [cm[:, 0], pm[:, 0]], [RED, GREEN],
                ["C.x raw", "P.x smooth"], 0, half, "camera path X (px)")
    _line_chart(canvas, [cm[:, 1], pm[:, 1]], [RED, GREEN],
                ["C.y raw", "P.y smooth"], half, H, "camera path Y (px)")
    return canvas


def plot_guard_timeline(levels, strength=None, size=(720, 160)) -> np.ndarray:
    """全局: 逐帧守门等级色带 + 防抖强度曲线."""
    W, H = size
    canvas = np.full((H, W, 3), 24, np.uint8)
    lv = np.array([int(l) for l in levels])
    T = len(lv)
    band_h = 40
    for t in range(T):
        x0 = int(t / T * W)
        x1 = int((t + 1) / T * W)
        cv2.rectangle(canvas, (x0, 24), (x1, 24 + band_h),
                      LEVEL_COLORS.get(lv[t], WHITE), -1)
    _label(canvas, "guard level per frame (green=L0 yellow=L1 red=L2)",
           org=(6, 18))
    if strength is not None and T > 1:
        _line_chart(canvas, [np.asarray(strength, np.float32)], [CYAN],
                    ["strength 0..1"], 24 + band_h, H, "stabilize strength")
    counts = {n: int((lv == k).mean() * 100)
              for k, n in LEVEL_NAMES.items()}
    _label(canvas, f"L0={counts['L0']}% L1={counts['L1']}% L2={counts['L2']}%",
           org=(W - 200, 18), color=CYAN)
    return canvas


def stack_grid(images, cols=2, pad=4, bg=15) -> np.ndarray:
    """把多张等尺寸图拼成网格面板."""
    if not images:
        return np.zeros((10, 10, 3), np.uint8)
    h, w = images[0].shape[:2]
    rows = (len(images) + cols - 1) // cols
    canvas = np.full((rows * h + (rows + 1) * pad,
                      cols * w + (cols + 1) * pad, 3), bg, np.uint8)
    for idx, im in enumerate(images):
        r, c = divmod(idx, cols)
        y = pad + r * (h + pad)
        x = pad + c * (w + pad)
        canvas[y:y + h, x:x + w] = im
    return canvas
