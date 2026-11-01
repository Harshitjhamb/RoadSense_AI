"""Generates the explanatory figures used in README.md and the project report.
Numbers come from the validation run in training/2_EvaluationTesting.ipynb and
from the Ultralytics YOLOv8 model table (COCO val, 640 px)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
NAVY, BLUE, PURPLE, GREY, RED, GREEN = "#1e3c72", "#2a5298", "#6a11cb", "#8a93ab", "#d9534f", "#2e9e5b"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

# 1 ── class distribution (validation set) ───────────────────────────────
names = ["Longitudinal\nCrack (D00)", "Transverse\nCrack (D10)", "Alligator\nCrack (D20)", "Pothole\n(D40)"]
counts = [555, 371, 848, 498]
fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(names, counts, color=[BLUE, PURPLE, NAVY, GREEN])
tot = sum(counts)
for b, c in zip(bars, counts):
    ax.text(b.get_x() + b.get_width() / 2, c + 12, f"{c}\n({c/tot:.0%})", ha="center", fontsize=10)
ax.set_ylabel("Labelled instances"); ax.set_ylim(0, 1000)
ax.set_title("Class distribution – validation set (1,294 images, 2,272 instances)")
fig.tight_layout(); fig.savefig(OUT / "class_distribution.png", dpi=170); plt.close(fig)

# 2 ── IoU explained ─────────────────────────────────────────────────────
fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.6))
cases = [((1, 1, 4, 4), (1.3, 1.3, 4, 4), "High overlap"),
         ((1, 1, 4, 4), (3, 3, 4, 4), "Partial overlap (miss)"),
         ((1, 1, 4, 4), (6, 6, 3, 3), "No overlap (miss)")]
def iou(a, b):
    ax1, ay1, aw, ah = a; bx1, by1, bw, bh = b
    ix = max(0, min(ax1 + aw, bx1 + bw) - max(ax1, bx1)); iy = max(0, min(ay1 + ah, by1 + bh) - max(ay1, by1))
    inter = ix * iy; return inter / (aw * ah + bw * bh - inter)
for ax, (gt, pr, t) in zip(axs, cases):
    ax.add_patch(Rectangle(gt[:2], gt[2], gt[3], fill=False, ec=GREEN, lw=2.5, label="Ground truth"))
    ax.add_patch(Rectangle(pr[:2], pr[2], pr[3], fill=False, ec=RED, lw=2.5, ls="--", label="Prediction"))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(f"{t.splitlines()[0]}\nIoU = {iou(gt, pr):.2f}", fontsize=10)
axs[0].legend(loc="lower right", fontsize=8)
fig.suptitle("Intersection over Union = area of overlap ÷ area of union  (a detection is a TP if IoU ≥ threshold, 0.5 for mAP@0.5)", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "iou_explained.png", dpi=170); plt.close(fig)

# 3 ── Pascal VOC → YOLO conversion (illustrative worked example) ─────────
fig, ax = plt.subplots(figsize=(10, 4.2)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 4.2)
ax.add_patch(Rectangle((0.2, 0.7), 3.6, 2.7, fill=False, ec=NAVY, lw=2)); ax.text(2, 3.55, "Image 600 × 400 px", ha="center", color=NAVY)
ax.add_patch(Rectangle((1.0, 1.2), 1.6, 1.0, fc="#fde2e1", ec=RED, lw=2)); ax.text(1.8, 1.7, "D40\npothole", ha="center", va="center", fontsize=9)
ax.text(4.0, 3.4, "Pascal VOC XML (RDD2022)", weight="bold", color=NAVY)
ax.text(4.0, 2.35, "<name>D40</name>\n<xmin>180</xmin>\n<ymin>150</ymin>\n<xmax>420</xmax>\n<ymax>300</ymax>", family="monospace", fontsize=8.5, va="center")
ax.annotate("", xy=(6.9, 2.4), xytext=(5.9, 2.4), arrowprops=dict(arrowstyle="->", lw=2, color=PURPLE))
ax.text(6.4, 2.7, "convert", ha="center", color=PURPLE, fontsize=9)
ax.text(7.1, 3.4, "YOLO .txt (one line / box)", weight="bold", color=NAVY)
ax.text(7.1, 2.35, "3 0.5000 0.5625 0.4000 0.3750", family="monospace", fontsize=9, va="center")
ax.text(0.2, 0.35, "w = 420−180 = 240 → 240/600 = 0.40      h = 300−150 = 150 → 150/400 = 0.375\n"
        "x_c = (180+120)/600 = 0.50       y_c = (150+75)/400 = 0.5625       class D40 → id 3",
        family="monospace", fontsize=8.5, va="center")
ax.set_title("Annotation conversion (illustrative example): pixel corners → normalised centre/size", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "voc_to_yolo.png", dpi=170); plt.close(fig)

# 4 ── block-diagram helper ───────────────────────────────────────────────
def boxes(ax, items, y, w, h, colors, gap=0.35, x0=0.3, fs=9):
    xs = []
    x = x0
    for (t, c) in zip(items, colors):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=c, ec="none"))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", color="white", fontsize=fs, weight="bold")
        xs.append(x); x += w + gap
    for a, b in zip(xs[:-1], xs[1:]):
        ax.add_patch(FancyArrowPatch((a + w, y + h / 2), (b, y + h / 2), arrowstyle="->", mutation_scale=14, lw=1.6, color=GREY))
    return xs

# 4a YOLOv8 architecture
fig, ax = plt.subplots(figsize=(11, 4)); ax.axis("off"); ax.set_xlim(0, 11.3); ax.set_ylim(0, 4)
boxes(ax, ["Input\n640×640×3", "Backbone\nCSPDarknet\n(Conv + C2f)", "SPPF", "Neck\nPAN-FPN\n(up/down-sample)", "Decoupled\nHead\n(3 scales)", "Boxes +\nclass scores\n→ NMS"],
      1.9, 1.5, 1.3, [GREY, BLUE, BLUE, PURPLE, NAVY, GREEN], gap=0.35, x0=0.25)
ax.text(5.65, 3.65, "YOLOv8s – single-stage, anchor-free detector", ha="center", weight="bold", color=NAVY, fontsize=12)
ax.text(5.65, 1.45, "Strides 8 / 16 / 32 → 80×80, 40×40, 20×20 grids  (small → large objects)", ha="center", fontsize=9, color=GREY)
ax.text(5.65, 0.85, "Head predicts box (x, y, w, h) and 4 class probabilities per grid cell – no anchor boxes, separate box and class branches", ha="center", fontsize=9)
ax.text(5.65, 0.35, "Loss = box (CIoU + DFL) + classification (BCE) · label assignment = Task-Aligned Assigner", ha="center", fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "yolov8_architecture.png", dpi=170); plt.close(fig)

# 4b inference pipelines (image + video)
fig, ax = plt.subplots(figsize=(11, 4.4)); ax.axis("off"); ax.set_xlim(0, 11.3); ax.set_ylim(0, 4.4)
ax.text(0.2, 4.1, "Image pipeline (pages/2_Image Detection.py)", weight="bold", color=NAVY)
boxes(ax, ["Upload\nPNG/JPG/\nWEBP/BMP", "EXIF rotate\n+ convert\nto RGB", "Resize\n640×640", "RGB→BGR\nYOLOv8s\npredict", "NMS +\nconf ≥ τ", "Draw boxes,\nresize back,\ntable + download"],
      2.55, 1.5, 1.2, [GREY, BLUE, BLUE, NAVY, PURPLE, GREEN], gap=0.35, x0=0.25, fs=8.5)
ax.text(0.2, 2.15, "Video pipeline (pages/3_Video Detection.py)", weight="bold", color=NAVY)
boxes(ax, ["Upload\n.mp4", "cv2.VideoCapture\nread frame", "Resize\n640×640", "YOLOv8s\npredict\n(per frame)", "Draw boxes,\nresize to\noriginal size", "VideoWriter\n(mp4v) →\ndownload"],
      0.55, 1.5, 1.2, [GREY, BLUE, BLUE, NAVY, PURPLE, GREEN], gap=0.35, x0=0.25, fs=8.5)
ax.text(5.65, 0.15, "↺ repeated for every frame; progress bar updated each iteration; τ = user-selected confidence threshold", ha="center", fontsize=9, color=GREY)
fig.tight_layout(); fig.savefig(OUT / "inference_pipelines.png", dpi=170); plt.close(fig)

# 5 ── model-size trade-off (Ultralytics published COCO numbers) ──────────
v = ["n", "s", "m", "l", "x"]; params = [3.2, 11.2, 25.9, 43.7, 68.2]; mapc = [37.3, 44.9, 50.2, 52.9, 53.9]; gflops = [8.7, 28.6, 78.9, 165.2, 257.8]
fig, ax = plt.subplots(figsize=(7.2, 4.2))
sc = ax.scatter(gflops, mapc, s=[p * 14 for p in params], c=[GREY, GREEN, BLUE, BLUE, BLUE], alpha=.85)
for a, b, n, p in zip(gflops, mapc, v, params):
    ax.annotate(f"YOLOv8{n}\n{p}M params", (a, b), textcoords="offset points", xytext=(10, -22 if n != "s" else 8), fontsize=9)
ax.set_xlabel("Compute (GFLOPs @ 640 px)"); ax.set_ylabel("COCO mAP@0.5:0.95")
ax.set_title("YOLOv8 variants: accuracy vs compute (bubble = parameters)\nbigger models give diminishing returns; 's' is the sweet spot", fontsize=10)
ax.set_xlim(-10, 300); ax.set_ylim(34, 57)
fig.tight_layout(); fig.savefig(OUT / "model_size_tradeoff.png", dpi=170); plt.close(fig)

# 6 ── metric cheat-sheet ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 3.4)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 3.4)
rows = [("Precision", "TP / (TP + FP)", "Of everything the model flagged, how much was really damage?"),
        ("Recall", "TP / (TP + FN)", "Of all real damage, how much did the model find?"),
        ("F1-score", "2·P·R / (P + R)", "Single number balancing precision and recall"),
        ("AP (per class)", "area under PR curve", "Precision averaged over all recall levels for one class"),
        ("mAP@0.5", "mean of AP at IoU 0.5", "Lenient: box only needs ≥50% overlap"),
        ("mAP@0.5:0.95", "mean over IoU 0.5…0.95", "Strict (COCO): rewards tight, well-localised boxes")]
for i, (a, b, c) in enumerate(rows):
    y = 3.0 - i * 0.52
    ax.add_patch(FancyBboxPatch((0.1, y - 0.18), 1.9, 0.38, boxstyle="round,pad=0.02", fc=BLUE, ec="none")); ax.text(1.05, y, a, ha="center", va="center", color="white", weight="bold", fontsize=10)
    ax.text(2.2, y, b, va="center", family="monospace", fontsize=9.5); ax.text(5.0, y, c, va="center", fontsize=9.5)
ax.set_title("Detection metrics at a glance", color=NAVY, weight="bold")
fig.tight_layout(); fig.savefig(OUT / "metrics_cheatsheet.png", dpi=170); plt.close(fig)
print("ok")
