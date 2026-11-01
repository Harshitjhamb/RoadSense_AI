# 🛣️ RoadSense AI – AI-Based Road Damage and Pothole Detection

> Detects and classifies road surface damage (**Longitudinal Crack, Transverse Crack, Alligator Crack, Pothole**) in **images and videos** using a **YOLOv8-small** object detector trained on the **India subset of the RDD2022 / CRDDC2022 dataset**, served through a **Streamlit** web app.

**Live demo:** https://roadsenseai01.streamlit.app/ (free tier – may be slow or sleeping; see [Deployment](#12-deployment))
**Full technical documentation:** [`02_DOCUMENTATION.md`](02_DOCUMENTATION.md) (architecture, code walkthrough, model internals, troubleshooting, how-to guides)
**Course:** B.Tech CSE, 5th semester, Machine Learning project (CSBB 311) – National Institute of Technology Delhi
**Supervisor:** Dr. Gautam Kumar



![Image Detection page of the deployed app](figures/app_image_detection.png){width=55%}



---

## Table of contents

1. [Project in one minute](#1-project-in-one-minute)
2. [Repository structure](#2-repository-structure)
3. [The problem and why it is object detection](#3-the-problem-and-why-it-is-object-detection)
4. [Dataset – RDD2022 / CRDDC2022](#4-dataset--rdd2022--crddc2022)
5. [Data preparation, step by step](#5-data-preparation-step-by-step)
6. [Data and ML concepts you must know](#6-data-and-ml-concepts-you-must-know)
7. [Choosing the model – why YOLOv8s](#7-choosing-the-model--why-yolov8s)
8. [Training](#8-training)
9. [Evaluation and results](#9-evaluation-and-results)
10. [Image analysis pipeline](#10-image-analysis-pipeline)
11. [Video analysis pipeline](#11-video-analysis-pipeline)
12. [Deployment](#12-deployment)
13. [Run it yourself](#13-run-it-yourself)
14. [Known limitations and future work](#14-known-limitations-and-future-work)
15. [**Interview / viva question bank**](#15-interview--viva-question-bank)
16. [Team, credits, licences](#16-team-credits-licences)

> ### ⚠️ Read before your viva – facts to verify with the team
> These are things a careful examiner could probe. Settle them as a team and keep your answers consistent.
> 1. **Weights provenance.** `models/YOLOv8_Small_RDD.pt` is the baseline weight file of the upstream open-source project [oracl4/RoadDamageDetection](https://github.com/oracl4/RoadDamageDetection) (the app even downloads it from that URL if the file is missing). Our contribution, as stated in this README, is to *study, re-run, evaluate and extend* that protocol on the India subset and to build the app. Do **not** claim to have trained those exact weights from scratch unless the team really did (the training run logs / `args.yaml` would prove it).
> 2. **Split sizes.** The saved output cells of `training/0_PrepareDatasetYOLOv8.ipynb` print `3594 / 399` (train / val) for India, while the evaluation run (`2_EvaluationTesting.ipynb`) validates on **1,294 images / 2,272 instances**. Those outputs come from different runs. Quote the evaluation figures (1,294 / 2,272) – they are the ones behind every result in this repo – and be ready to say the prep-notebook output is from an earlier run of the upstream notebook.
> 3. **Video page channel order.** `pages/3_Video Detection.py` converts frames BGR→RGB *before* `predict`, but Ultralytics expects BGR numpy arrays (the image page was fixed, the video page was not). The output video looks right, but the model sees colour-swapped frames, which can slightly hurt accuracy. See [limitations](#14-known-limitations-and-future-work).

---

## 1. Project in one minute

| | |
|---|---|
| **Input** | A road photo (PNG/JPG/JPEG/WEBP/BMP) or an `.mp4` video |
| **Output** | The same image/video with a coloured box, class name and confidence around every damaged region; a table of detections; a download button |
| **Classes (4)** | Longitudinal Crack (D00), Transverse Crack (D10), Alligator Crack (D20), Pothole (D40) |
| **Data** | RDD2022 – **India** subset (7,706 training images, Pascal VOC XML) |
| **Model** | YOLOv8s (single-stage, anchor-free), 11.1 M parameters, 28.4 GFLOPs, fine-tuned from COCO weights |
| **Result** | Precision 0.584 · Recall 0.523 · **mAP@0.5 0.547** · mAP@0.5:0.95 0.254 on 1,294 validation images |
| **Speed** | ≈ 13 ms/image on an RTX 2060 GPU (~75 FPS); ≈ 160 ms/image on the free cloud CPU |
| **App** | Streamlit multi-page app (Home · Image · Video · Realtime "coming soon"), deployed on Streamlit Community Cloud |

**The ML life cycle followed:**



![Overall pipeline](figures/pipeline.png){width=95%}



`Collect data → convert annotations → clean/filter → split → choose model → train → evaluate → build app → deploy`

---

## 2. Repository structure

```
RoadDamageDetection/
├── Home.py                      # Landing page (Streamlit entry point)
├── pages/
│   ├── 1_Realtime Detection.py  # "Coming soon" placeholder
│   ├── 2_Image Detection.py     # Image upload → detection → table + download
│   └── 3_Video Detection.py     # .mp4 upload → frame-by-frame detection → download
├── models/YOLOv8_Small_RDD.pt   # Trained YOLOv8s weights (~85 MB)
├── training/
│   ├── 0_PrepareDatasetYOLOv8.ipynb   # VOC XML → YOLO txt, filter classes, split
│   ├── 1_TrainingYOLOv8.ipynb         # Train YOLOv8s
│   ├── 2_EvaluationTesting.ipynb      # Validate: P, R, mAP, confusion matrix
│   ├── dataset/rddIndiaFiltered/rdd_India.yaml   # dataset config
│   ├── tools/make_figures.py          # regenerates figures/*.png
│   └── yolov8n.pt, yolov8s.pt         # COCO pre-trained starting weights
├── sample_utils/download.py     # downloads weights if missing
├── figures/                    # PR curve, confusion matrix, validation batch images, demo GIFs
├── docs/
│   ├── ML Project Report.docx   # Full project report (with figures)
│   └── figures/                 # Diagrams and screenshots used here and in the report
├── requirements.txt             # pinned Python dependencies
├── packages.txt                 # apt packages for Streamlit Cloud (libgl1, libglib2.0-0)
└── .streamlit/config.toml       # 1 GB upload limit, light theme
```

---

## 3. The problem and why it is object detection

Road inspection in India is mostly manual: slow, subjective, expensive, and large parts of the road network are rarely surveyed. Unrepaired cracks and potholes grow every monsoon and cause accidents and vehicle damage. A smartphone camera plus a trained model can make inspection **faster, cheaper, consistent and scalable**.

**Which ML task is this?**

| Task | Output | Fits? |
|---|---|---|
| Image classification | One label per image ("this image has a pothole") | ❌ A frame can hold several damages of different types; engineers need to know *where* |
| **Object detection** | **Box + class + confidence for every object** | ✅ Exactly what we need |
| Semantic/instance segmentation | A class (or mask) for every pixel | ➖ More precise (crack width/area) but needs expensive pixel-level labels; RDD2022 only has boxes |

So the problem is **supervised, multi-class object detection** with four classes. The hard parts: cracks are thin and low-contrast and easily confused with shadows, road markings, joints and patches; potholes change appearance with lighting, water and size; the model must be fast enough for video; and non-experts must be able to use it.

---

## 4. Dataset – RDD2022 / CRDDC2022

**RDD2022** is the dataset of the *Crowdsensing-based Road Damage Detection Challenge 2022* (IEEE Big Data Cup 2022). It contains about **47,420 smartphone road images from six countries** (Japan, India, Czech Republic, Norway, United States, China – the latter split into MotorBike and Drone sub-sets) with more than 55,000 annotated damage instances. Images are collected "crowdsensing" style – ordinary cameras on ordinary vehicles – which makes it realistic for a low-cost system like ours.

Number of **training** images per sub-set (from the notebook output):

| Japan | India | China-Drone | China-MotorBike | Czech | Norway | United States |
|---|---|---|---|---|---|---|
| 10,506 | **7,706** | 2,401 | 1,977 | 2,829 | 8,161 | 4,805 |

**Annotation format:** one **Pascal VOC XML** file per image with `<size>` (width/height) and one `<object>` per damage with `<name>` (damage code) and `<bndbox>` (`xmin, ymin, xmax, ymax` in pixels).

**Damage codes used (the 4 official CRDDC classes):**

| Class ID | Code | Name | What it is |
|---|---|---|---|
| 0 | D00 | Longitudinal Crack | Crack running parallel to the road direction |
| 1 | D10 | Transverse Crack | Crack running across the road |
| 2 | D20 | Alligator Crack | Interconnected cracks forming a mesh (fatigue failure) |
| 3 | D40 | Pothole | Bowl-shaped cavity where surface material has broken away |

Other codes can appear in annotation files (D01, D11 construction joints; D43 crosswalk blur; D44 white-line blur; D50 manhole/utility hole). They are **not our task**, so the conversion script drops them.

**Why only India?**
- Road surfaces, markings, traffic, lighting and camera mounting differ per country. Training and testing on the **target domain** (Indian roads) gives a model that matches the use case and avoids domain-shift noise from, e.g., Norwegian winter roads or Japanese roads.
- A single-country set trains faster on a 6 GB GPU and keeps the experiment clean and explainable.
- Trade-off: less data than all six countries combined (see limitations).

**Why no test set?** The official RDD2022 *test* annotations are not public (they were used for the challenge leaderboard). We therefore split the labelled training images into **train/validation** ourselves.

**Class balance (validation set, 2,272 instances):**



![Class distribution](figures/class_distribution.png){width=55%}



Alligator cracks are the most frequent (37%), transverse cracks the rarest (16%). Imbalance is one reason transverse cracks score lowest.

---

## 5. Data preparation, step by step

Implemented in `training/0_PrepareDatasetYOLOv8.ipynb`.

### 5.1 Convert Pascal VOC → YOLO format

YOLO wants one `.txt` per image, one line per box: `class x_center y_center width height`, all **normalised to 0–1** by the image size.



![VOC to YOLO conversion](figures/voc_to_yolo.png){width=90%}



```
w   = xmax − xmin               h   = ymax − ymin
cx  = xmin + w/2                cy  = ymin + h/2
w_n = w / image_width           h_n = h / image_height
cx_n = cx / image_width         cy_n = cy / image_height
```

Why normalise? Labels become independent of image resolution, so the same file works when the image is resized to 640 × 640 for training.

The function `convertPascal2YOLOv8` maps the codes (`D00→0, D10→1, D20→2, D40→3`, anything else → "ignored") and writes **only classes < 4**.

### 5.2 Background images (negative samples)
Images with no usable damage are kept only up to **10 % of the dataset size**. Why? A model that never sees "nothing to detect" produces false alarms on clean roads; but too many empty images would swamp the positives and push the model toward predicting nothing. 10 % is a compromise.

### 5.3 Train / validation split
`CopyDatasetSplit` shuffles with **`random.seed(1337)`** and splits **90 % train / 10 % validation**, copying images and labels to `images/{train,val}` and `labels/{train,val}`. The fixed seed makes the split **reproducible**. The dataset YAML (`rdd_India.yaml`) lists the train/val folders and the 4 class names.

> **Caveat to mention honestly:** random image-level splitting can put near-duplicate frames from the same road/video in both sets (data leakage), which makes validation scores a little optimistic. A split by road/sequence would be stricter.

### 5.4 Resulting evaluation set
`1,294 images (177 background), 2,272 labelled instances`: Longitudinal 555 · Transverse 371 · Alligator 848 · Pothole 498.

---

## 6. Data and ML concepts you must know

Short, exam-ready definitions grouped by topic. (Every one of these is a possible question.)

### 6.1 Data concepts
- **Supervised learning** – learning from labelled examples (image + boxes + classes).
- **Bounding box** – rectangle around an object; stored as corners (VOC) or centre/size (YOLO).
- **Annotation / label noise** – humans label thin cracks inconsistently; this caps the achievable accuracy.
- **Normalisation** – scaling coordinates to 0–1 (labels) and pixel values to 0–1 (images) so training is stable and resolution-independent.
- **Train / validation / test** – *train* fits the weights; *validation* tunes choices and monitors overfitting; *test* is touched once for the final unbiased number. We have no public test set, so the validation split doubles as our report set.
- **Data leakage** – information from validation leaking into training (e.g. near-duplicate frames) → over-optimistic scores.
- **Class imbalance** – unequal class frequencies; rare classes are learned worse. Remedies: more data, class weights, oversampling, focal loss, targeted augmentation.
- **Background / negative images** – images without objects; reduce false positives.
- **Data augmentation** – random label-preserving changes (flip, scale, translate, HSV colour jitter, mosaic) that enlarge the effective dataset and improve generalisation.
- **Mosaic augmentation** – stitches 4 training images into one. We **switched it off** (`mosaic=0.0`) because stitching cuts and rescales thin cracks and creates artificial borders that look like cracks.
- **Domain shift** – train and deployment data differ (country, camera, weather) → accuracy drops.
- **Transfer learning** – start from weights trained on a big dataset (COCO, 80 everyday classes) and fine-tune. Early layers already know edges/textures; we only adapt to roads. Faster, needs less data, usually more accurate.

### 6.2 Model/training concepts
- **Epoch** – one full pass over the training set (we use 100). **Batch size** – images per weight update (32). **Iteration** – one update.
- **Learning rate (LR)** – step size of gradient descent. **Warm-up** – start with a tiny LR for the first 5 epochs so fresh heads don't destroy pre-trained features. **Cosine schedule** – LR decays smoothly like a half-cosine, so late training takes small, stable steps.
- **Overfitting** – train score ≫ validation score (model memorises). **Underfitting** – both low. Signs: watch val loss/mAP curves; fixes: more data, augmentation, early stopping, smaller model, weight decay.
- **Loss** – the number being minimised. YOLOv8 uses **box loss** (CIoU + Distribution Focal Loss), **classification loss** (binary cross-entropy) – there is no separate objectness term (anchor-free).
- **Seed** – fixing the random seed (1337) makes runs reproducible (GPU non-determinism aside).
- **Checkpoint** (`save_period=10`) – saved weights every 10 epochs so training can be resumed.
- **Anchor boxes** – pre-defined box shapes older detectors regress from. YOLOv8 is **anchor-free**: it predicts the box directly from each grid location – fewer hyper-parameters, better with odd shapes like long thin cracks.
- **Stride / feature map** – the network predicts at strides 8, 16, 32 (80×80, 40×40, 20×20 grids for a 640 input) to catch small, medium, large objects.
- **NMS (non-maximum suppression)** – many overlapping boxes are predicted for one object; NMS keeps the highest-confidence box and removes others whose IoU with it exceeds a threshold.

### 6.3 Evaluation concepts



![IoU](figures/iou_explained.png){width=85%}



- **IoU** = overlap area ÷ union area of predicted and true box.
- **TP** – correct class and IoU ≥ threshold. **FP** – a box with no matching true damage (or wrong class). **FN** – real damage that no box matched. **TN** – not meaningful in detection (infinite empty boxes).
- **Precision** = TP/(TP+FP) – *of what I flagged, how much was real?* **Recall** = TP/(TP+FN) – *of what is real, how much did I find?* **F1** = 2PR/(P+R).
- **Confidence threshold (τ)** – minimum score to display a box. Low τ → high recall, low precision; high τ → opposite. This is why the app has a slider.
- **PR curve** – precision vs recall as τ sweeps; **AP** = area under it for one class; **mAP** = mean AP over classes.
- **mAP@0.5** – IoU threshold 0.5 (lenient). **mAP@0.5:0.95** – averaged over IoU 0.5, 0.55 … 0.95 (strict, the COCO metric; rewards tight boxes).
- **Confusion matrix** – rows/cols = classes + "background"; shows class mix-ups, missed objects (FN → predicted as background) and false alarms (FP on background).



![Metrics cheat sheet](figures/metrics_cheatsheet.png){width=85%}



---

## 7. Choosing the model – why YOLOv8s

### 7.1 Requirements
1. Must **localise and classify** multiple damages per image → object detection.
2. Must be **fast** (video, ideally real-time) → single-stage.
3. Must be **accurate on small/thin objects** → multi-scale features.
4. Must be **trainable on one 6 GB consumer GPU** and **servable on a free CPU host** → small model.
5. Must have **pre-trained weights and a mature tool-chain** (train, validate, export, plot) → open source, active.

### 7.2 Alternatives considered (design rationale – we did not train these)

| Family | Strengths | Why not chosen |
|---|---|---|
| **Faster R-CNN** (two-stage) | High accuracy, strong baseline | Slower and heavier; harder to serve on CPU/video |
| **SSD / RetinaNet** | Fast single-stage | Anchor tuning needed; SSD weaker on small objects; older tool-chains |
| **YOLOv5** | Mature, popular | Anchor-based; YOLOv8 is its successor with better accuracy/speed |
| **DETR / RT-DETR** (transformers) | No NMS, global context | Long training schedules, more data and GPU memory; less friendly for a small project |
| **Segmentation (Mask R-CNN, YOLOv8-seg)** | Pixel-accurate masks | Needs pixel labels; RDD2022 only has boxes |
| **YOLOv8** ✅ | Single-stage, anchor-free, fast, strong accuracy, COCO weights, full tool-chain | – |

### 7.3 Why the *small* variant (`s`)?



![YOLOv8 sizes](figures/model_size_tradeoff.png){width=60%}



| Variant | Params | GFLOPs | COCO mAP@0.5:0.95 |
|---|---|---|---|
| YOLOv8n | 3.2 M | 8.7 | 37.3 |
| **YOLOv8s** | **11.2 M** | **28.6** | **44.9** |
| YOLOv8m | 25.9 M | 78.9 | 50.2 |
| YOLOv8l | 43.7 M | 165.2 | 52.9 |
| YOLOv8x | 68.2 M | 257.8 | 53.9 |

(Published Ultralytics COCO numbers; our road model: 11.1 M params, 28.4 GFLOPs with 4 classes.)
`n` loses too much accuracy for thin cracks; `m` and above need ~2.3×–6× the parameters and ≥2.8× the compute for modest gains and would not fit comfortably on a free CPU host; `s` is the **knee of the curve**. A larger model is listed as future work.

### 7.4 How YOLOv8 works (30-second version)



![YOLOv8 architecture](figures/yolov8_architecture.png){width=90%}



1. **Backbone (CSPDarknet with C2f blocks + SPPF)** – turns the image into feature maps at several scales (edges → textures → shapes).
2. **Neck (PAN-FPN)** – mixes high-resolution detail with deep semantic features (top-down and bottom-up paths) so small cracks and big potholes are both seen.
3. **Decoupled head** – separate branches predict *where* (box) and *what* (class) at each grid cell of the three scales; **no anchors**.
4. **Training-time label assignment** – the **Task-Aligned Assigner** decides which grid cells are responsible for which ground-truth box.
5. **Inference** – every cell proposes a box + class scores; low-confidence ones are dropped (τ) and **NMS** removes duplicates.

"You Only Look Once": the whole image goes through the network **once**, instead of proposing regions first and classifying them (two-stage).

---

## 8. Training

Implemented in `training/1_TrainingYOLOv8.ipynb`.

```python
model = YOLO('yolov8s.pt')                  # COCO pre-trained
model.train(data='rdd_India.yaml', epochs=100, warmup_epochs=5,
            batch=32, imgsz=640, save_period=10, workers=1,
            seed=1337, cos_lr=True, mosaic=0.0)
```

| Setting | Value | Why |
|---|---|---|
| Base weights | `yolov8s.pt` (COCO) | Transfer learning |
| Epochs / warm-up | 100 / 5 (5 %) | Enough to converge on ~thousands of images; warm-up protects pre-trained features |
| Batch size | 32 | Fits a 6 GB GPU at 640 px; stable gradients |
| Image size | 640 × 640 | YOLO standard; cracks stay visible; speed/accuracy balance |
| LR schedule | Cosine (`cos_lr=True`) | Smooth decay, good final convergence |
| Mosaic | **Off (0.0)** | Preserves thin cracks (see §6.1) |
| Seed | 1337 | Reproducibility |
| Checkpoints | every 10 epochs | Resume with `resume=True` |
| Other | Ultralytics defaults (optimizer `auto`, HSV colour jitter, scale/translate, horizontal flip, weight decay) | Not changed |
| Hardware | NVIDIA RTX 2060 (6 GB), CUDA, PyTorch 2.0, Ultralytics 8.0.201 | |

`resume=True` continues an interrupted run from `last.pt`.

---

## 9. Evaluation and results

`training/2_EvaluationTesting.ipynb` runs `model.val()` on the validation split.

**Validation set: 1,294 images, 2,272 instances** (≈ 13 ms/img on RTX 2060: 0.7 pre + 9.3 inference + 3.4 post).

| Class | Instances | Precision | Recall | F1 | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|---|---|
| Longitudinal Crack | 555 | 0.573 | 0.471 | 0.517 | 0.501 | 0.235 |
| Transverse Crack | 371 | 0.522 | 0.433 | 0.473 | 0.454 | 0.177 |
| Alligator Crack | 848 | 0.659 | 0.684 | 0.671 | **0.709** | 0.373 |
| Pothole | 498 | 0.584 | 0.503 | 0.540 | 0.524 | 0.229 |
| **All** | 2,272 | **0.584** | **0.523** | **0.552** | **0.547** | **0.254** |



![Per-class metrics](figures/per_class_metrics.png){width=60%}



    

![](figures/PR_curve.png){width=48%}


    

![](figures/confusion_matrix.png){width=48%}





![Predictions on a validation batch](figures/val_batch2_pred.jpg){width=70%}



### How to read the results
- **Alligator cracks are best (0.709 mAP@0.5):** large, textured, distinctive, and the most frequent class.
- **Transverse cracks are worst (0.454):** thin, short, few pixels at 640 px, easily confused with joints/edges/shadows, and the rarest class.
- **Classes are rarely confused with each other.** In the confusion matrix only a handful of cases go between damage types (e.g. 9 longitudinal → alligator, 17 alligator → longitudinal). The errors are mostly **missed damage** (≈ 190–253 per class predicted as background) and **false alarms on background** (≈ 133–224 per class).
- **PR curve:** precision stays high (> ~0.8) up to ~0.3 recall, then falls – so raising τ gives cleaner but fewer detections.
- **mAP@0.5:0.95 (0.254) ≪ mAP@0.5 (0.547):** the model finds damage but boxes are not always tight – expected, because crack extents are ambiguous even for human labellers.
- **Is 0.547 good?** For this dataset, yes, it is a reasonable baseline: road damage is a hard, noisy-label problem. It is *not* production grade (about half of the thin cracks are missed) – we are explicit about that.

---

## 10. Image analysis pipeline

File: `pages/2_Image Detection.py` · diagram below (top row).



![Inference pipelines](figures/inference_pipelines.png){width=95%}



1. **Upload** (`st.file_uploader`: PNG, JPG, JPEG, WEBP, BMP).
2. **Normalise:** `ImageOps.exif_transpose` (phones store rotation in EXIF – without this the model sees a sideways road) → `.convert("RGB")` (handles grayscale, RGBA/PNG transparency, palette images).
3. **Resize** to 640 × 640 with `cv2.INTER_AREA` (good for down-scaling) – matches the training resolution.
4. **Colour order:** convert RGB → **BGR** because Ultralytics treats numpy arrays as BGR (OpenCV convention).
5. **Inference:** `net.predict(img, conf=τ)` – the network produces boxes; confidence filter τ and NMS are applied. Default **τ = 0.25** (lowered from 0.50 because confidence on thin cracks is naturally modest).
6. **Post-process:** `results[0].plot()` draws boxes/labels → BGR→RGB → **resize back to the original size** so the output has the user's resolution.
7. **Present:** original and annotated side by side, a **table** (class + confidence, sorted), a warning if nothing was detected ("lower the threshold"), and a **PNG download**.
8. **Model loading:** `YOLO(...)` is created once and stored in `st.session_state` so it isn't reloaded on every interaction. `download_file` fetches the weights if absent.

Fixes the team made to this pipeline: RGB normalisation, EXIF rotation, consistent RGB/BGR, more formats, default τ = 0.25, detection summary table.

---

## 11. Video analysis pipeline

File: `pages/3_Video Detection.py` · diagram above (bottom row).

1. **Upload** an `.mp4`; the file is written to `./temp/video_input.mp4`.
2. `cv2.VideoCapture` reads metadata (width, height, FPS, frame count → duration).
3. For **every frame**: resize to 640 × 640 → `net.predict(conf=τ)` (default **τ = 0.5** on this page) → draw boxes → resize back to the original width/height → write with `cv2.VideoWriter` (`mp4v` codec, same FPS) to `./temp/video_infer.mp4`; the current frame is shown live and a progress bar is updated.
4. When finished: **download button** for the annotated video and a restart button.

Design notes:
- **Frame-by-frame, independent detection** – no tracking, so the same pothole can get a new detection each frame and cannot be counted uniquely. Tracking (ByteTrack/BoT-SORT, built into Ultralytics) + counting is a natural extension.
- **`mp4v` codec** because pip-installed OpenCV has no H.264 (licensing); browsers often can't play `mp4v` inline, hence the download button.
- **Speed:** ≈ 13 ms/frame on GPU; ≈ 160 ms/frame on the free cloud CPU – long videos take minutes. Throttle: cloud CPU may be limited further.
- **Realtime (webcam)** is deliberately a "coming soon" page: it needs `streamlit-webrtc` + a TURN server on cloud hosting and further latency evaluation.

---

## 12. Deployment

Hosted on **Streamlit Community Cloud** (free), straight from this GitHub repo: main file `Home.py`, Python 3.11.

| File | Purpose |
|---|---|
| `requirements.txt` | Pinned pip dependencies; first line `--extra-index-url https://download.pytorch.org/whl/cpu` installs the **CPU-only** PyTorch build (smaller, no CUDA) |
| `packages.txt` | apt packages needed by OpenCV: `libgl1`, `libglib2.0-0` |
| `.streamlit/config.toml` | `maxUploadSize = 1000` MB, light theme |

**Real problems hit during deployment (good interview material):**

| Symptom | Cause | Fix |
|---|---|---|
| `Package 'libgl1-mesa-glx' has no installation candidate` | Streamlit Cloud moved to Debian *trixie*, which dropped that package | Use `libgl1` (+ `libglib2.0-0`) in `packages.txt` |
| `av==10.0.0` fails to install | No pre-built wheel for Python 3.11/3.12 → tries to compile PyAV, needs ffmpeg dev libs | Pin `av==12.3.0` (has wheels) |
| Large/slow install, memory pressure | Default PyTorch wheel bundles CUDA | CPU wheel index line |
| *"our app … we've temporarily reduced its CPU"* | Free-tier fair-use throttling of CPU-heavy apps (YOLO on CPU) | Wait for expiry; lower frame rate/size; or move to Hugging Face Spaces |
| pip warning: `streamlit 1.28.0 requires rich<14` | A transitive package pulled `rich 15` | Harmless so far; pin `rich<14` if it ever breaks |

Free-tier limits: ~1 GB RAM, shared CPU, app sleeps when idle. Alternatives: Hugging Face Spaces (more RAM), Render/Railway/Docker on a VM, or a GPU host for real-time use.

---

## 13. Run it yourself

```bash
git clone https://github.com/Harshitjhamb/RoadSense_AI.git
cd RoadSense_AI
python -m venv .venv && source .venv/bin/activate        # Python 3.10/3.11
pip install -r requirements.txt                           # pulls CPU PyTorch
streamlit run Home.py                                     # http://localhost:8501
```
(On Linux you may also need `sudo apt install libgl1 libglib2.0-0`. GPU users: install the CUDA build of PyTorch from pytorch.org first.)

**Retrain from scratch**
1. Download RDD2022 from the [RoadDamageDetector](https://github.com/sekilab/RoadDamageDetector) page and keep the `India` folder.
2. Edit the `ROOTDIR` / `baseOutputDir` paths in `training/0_PrepareDatasetYOLOv8.ipynb` and run it.
3. Run `1_TrainingYOLOv8.ipynb` (≈ hours on an RTX 2060), then `2_EvaluationTesting.ipynb`.
4. Regenerate the explanatory figures: `python training/tools/make_figures.py`.

---

## 14. Known limitations and future work

**Limitations (be upfront about them in a viva):**
- **Moderate accuracy on thin cracks** (mAP@0.5 0.45–0.50); roughly half of thin cracks are missed at the default operating point.
- **No independent test set** – scores are from a 10 % validation split of the training images; random splitting may leak near-duplicates → slightly optimistic.
- **Domain/device bias** – Indian smartphone images only.
- **Video page channel order** – frames are passed as RGB although Ultralytics expects BGR; output colours look right but the model sees swapped channels (the Image page does it correctly). Fix: remove the BGR→RGB conversion before `predict` (and adjust the later conversions).
- **Aspect-ratio squashing** – images are resized to 640 × 640 without padding (letterbox), slightly distorting wide frames. Passing the original image and `imgsz=640` lets Ultralytics letterbox correctly.
- **Shared temp files** – `./temp/video_input.mp4` / `video_infer.mp4` are fixed names, so two simultaneous users would overwrite each other.
- **No severity or GPS**, no tracking/counting in video, no realtime page.
- **Free-hosting limits** – slow, throttled, ~1 GB RAM.

**Future work:** realtime webcam / in-vehicle detection (WebRTC + TURN or edge device) · GPS-tagged map of damage · severity estimation (size/depth, PCI-style score) · bigger model (YOLOv8m), higher resolution (960/1280), crack-friendly augmentation and class re-balancing · add other countries for generalisation · segmentation masks for crack area · object tracking to count unique potholes · ONNX/TensorRT/TFLite export for phones and Jetson · maintenance dashboard and reports · proper held-out test set and split-by-road evaluation.

---

## 15. Interview / viva question bank

Answers are written to be spoken in 20–60 seconds. Where a number is quoted it comes from this repo's evaluation.

### A. The project (opening questions)

**Q1. Explain your project in 30 seconds.**
We built RoadSense AI, a system that detects and classifies road damage – longitudinal, transverse, alligator cracks and potholes – in images and videos. We trained a YOLOv8-small detector on the India subset of the RDD2022 dataset, evaluated it (mAP@0.5 = 0.547 on 1,294 validation images, ~13 ms/image on a GPU) and deployed it as a Streamlit web app where a user uploads an image or video and gets annotated results to download.

**Q2. What problem does it solve and who would use it?**
Manual road inspection is slow, subjective and expensive, so many roads go unsurveyed. A camera-only system lets municipal bodies, contractors, fleet operators or even citizens screen roads cheaply and consistently and prioritise repairs before damage grows.

**Q3. What is novel / what did *you* contribute?**
The project builds on the open-source oracl4 RoadDamageDetection baseline. Our contributions: restricting and re-running the protocol on the India subset, re-evaluating it, fixing the image pipeline (RGB normalisation, EXIF rotation, RGB/BGR consistency, extra formats, threshold 0.25, detection table), redesigning the app, writing the report/documentation and deploying it to the cloud. *(Align this answer with what your team actually did – see the verification box at the top.)*

**Q4. Why object detection and not classification or segmentation?**
One image can contain several damages of different types and engineers need locations. Classification gives only one label per image. Segmentation would give crack area but needs pixel-level labels; our dataset has bounding boxes only.

**Q5. Which technologies did you use?**
Python, PyTorch 2.0, Ultralytics YOLOv8 (8.0.201), OpenCV (headless), NumPy, Pillow, Streamlit, Jupyter, BeautifulSoup + tqdm for data prep; GitHub + Streamlit Community Cloud for hosting; RTX 2060 for training.

**Q6. Show me the end-to-end flow.**
Data (RDD2022 India) → VOC→YOLO conversion and class filter → background retention and 90/10 split → YOLOv8s fine-tuning → validation metrics → weights in `models/` → Streamlit pages that load the weights and run `predict` on uploaded images/video → annotated output.

### B. Dataset and data concepts

**Q7. What is RDD2022 and why did you choose it?**
A public multi-country dataset (~47k smartphone images, six countries, 55k+ damage boxes) from the Crowdsensing-based Road Damage Detection Challenge 2022. It is the largest open, consistently annotated road-damage dataset, includes India (our target), uses standard damage codes and has a clear licence for research.

**Q8. Why only the India subset?**
It matches our deployment domain, avoids domain shift from other countries' roads and cameras, makes training feasible on a 6 GB GPU, and keeps the study focused. The cost is less data; adding other countries is future work.

**Q9. What do D00, D10, D20, D40 mean?**
D00 longitudinal crack (along the road), D10 transverse crack (across), D20 alligator crack (interconnected mesh – fatigue cracking), D40 pothole (bowl-shaped cavity). Other codes (D01, D11, D43, D44, D50) are dropped.

**Q10. What format are the annotations and how did you convert them?**
Pascal VOC XML with pixel corner boxes. YOLO needs `class cx cy w h` normalised to 0–1. We compute width/height from corners, the centre, divide by image width/height, and write one `.txt` per image (`convertPascal2YOLOv8`). Example: box (180,150)-(420,300) in a 600×400 image → `3 0.5 0.5625 0.4 0.375`.

**Q11. Why normalise the coordinates?**
So labels don't depend on image resolution; the same label file stays valid when the image is resized to 640×640 or augmented.

**Q12. Why did you keep some background images and why only 10 %?**
Negative examples teach the model that "no damage" is a valid answer and reduce false positives on clean roads. Too many would dominate the loss and bias the model to predict nothing, so we cap them at 10 % of the dataset.

**Q13. How did you split the data? Why 90/10 and a seed?**
Random shuffle with seed 1337, 90 % train / 10 % validation. 90/10 keeps most data for learning while leaving ~1.3k images for a stable estimate; the seed makes the split reproducible. We had no public test set because RDD2022 test labels are withheld.

**Q14. Is there any data leakage? What would you do better?**
Possibly: consecutive frames of the same road may land in both sets, making scores optimistic. A better protocol splits by road/video sequence and keeps a truly separate test set.

**Q15. Is the data imbalanced? How did you handle it?**
Yes: alligator 37 %, longitudinal 24 %, pothole 22 %, transverse 16 % of validation instances. We did not re-weight; the effect shows up as lower accuracy on transverse cracks. Remedies: class weights, oversampling, targeted augmentation, more data.

**Q16. What preprocessing/augmentation did you apply?**
Offline: format conversion, class filtering, background control. Online (training): Ultralytics defaults (HSV colour jitter, scale, translate, horizontal flip) with **mosaic disabled**. At inference: RGB conversion, EXIF fix, resize to 640×640.

**Q17. Why did you disable mosaic augmentation?**
Mosaic stitches four images and rescales/crops them; for thin cracks this can cut a crack or create synthetic edges that look like cracks, harming precision. We prioritised label fidelity over augmentation strength.

**Q18. What is label noise and does it affect you?**
Annotators disagree about where a crack starts/ends and whether faint cracks exist. It limits attainable mAP@0.5:0.95 and explains why tight-box accuracy (0.254) is much lower than loose-box accuracy (0.547).

**Q19. What is domain shift?**
When deployment data differs from training data (new country, camera, lighting). Our model is trained on Indian smartphone images and may degrade elsewhere.

**Q20. Why is the test set missing and is your evaluation valid?**
The challenge's test annotations are not public. Our numbers are validation-set results; they are valid for comparing configurations and giving an estimate, but a bit optimistic versus completely unseen roads. We state this openly.

### C. Model choice

**Q21. Why YOLOv8 and not Faster R-CNN, SSD, RetinaNet or DETR?**
We need speed for video and a model small enough for a free CPU host. Faster R-CNN (two-stage) is more accurate in some settings but slower and heavier; SSD is weaker on small objects; RetinaNet needs anchor tuning; DETR needs long training and more data. YOLOv8 is single-stage, anchor-free, ships COCO weights and a full train/val/export pipeline. *(These are design rationales; we did not train the others.)*

**Q22. Why the small variant?**
It is the knee of the accuracy/compute curve: ~11 M parameters, 28 GFLOPs, COCO mAP 44.9. Nano loses too much accuracy; medium costs ~2.3× parameters and ~2.8× compute for a modest gain and strains a free host. It trains comfortably on a 6 GB GPU.

**Q23. Explain how YOLO works.**
The image is resized and passed once through a CNN (backbone → neck → head). The head predicts, for each cell of three grids (80×80, 40×40, 20×20), a box and class scores. Low-confidence predictions are removed and NMS deletes duplicates. One pass = very fast.

**Q24. What is the difference between one-stage and two-stage detectors?**
Two-stage (Faster R-CNN) first proposes regions, then classifies/refines each – accurate, slower. One-stage (YOLO, SSD) predicts boxes and classes directly in one pass – faster, historically a bit less accurate, now nearly equal.

**Q25. What does "anchor-free" mean and why does it help?**
Older detectors regress offsets from predefined anchor boxes whose sizes/ratios must be tuned. YOLOv8 predicts the box directly from each location (distances to box edges), removing anchor hyper-parameters and handling unusual shapes like long thin cracks more naturally.

**Q26. What are the backbone, neck and head?**
Backbone: CSPDarknet with C2f blocks and SPPF – extracts features. Neck: PAN-FPN – merges multi-scale features. Head: decoupled – separate branches for box regression and classification.

**Q27. What loss functions does YOLOv8 use?**
Box loss = CIoU + Distribution Focal Loss; classification loss = binary cross-entropy; positives are chosen by the Task-Aligned Assigner. No objectness branch.

**Q28. What is transfer learning and why use it here?**
Initialising from COCO-trained weights. Low-level features (edges, textures) transfer to roads, so we converge faster and need less data than training from scratch.

**Q29. Why not train from scratch or use a bigger model?**
From scratch needs far more data/compute and usually ends worse. A bigger model would be slower, riskier to overfit and heavier to deploy; it is future work.

**Q30. Model size and parameters?**
11,127,132 parameters, 168 layers after fusing Conv+BN, 28.4 GFLOPs, weight file ≈ 85 MB.

### D. Training

**Q31. List your training hyper-parameters and justify them.**
100 epochs, 5 warm-up epochs, batch 32, 640×640, cosine LR, mosaic off, seed 1337, checkpoints every 10 epochs, workers 1. Batch 32 fits 6 GB at 640 px; warm-up protects pretrained features; cosine decay gives smooth convergence; mosaic off protects thin cracks.

**Q32. What is an epoch, batch and iteration?**
Epoch: one pass over all training images. Batch: images per update (32). Iteration: one weight update (≈ train images / 32 per epoch).

**Q33. What is the learning rate, warm-up and cosine schedule?**
LR is the gradient-descent step size. Warm-up starts it very small and ramps up over the first 5 epochs to avoid wrecking pretrained weights. Cosine decay lowers it smoothly to near zero for stable final convergence.

**Q34. How do you detect overfitting and what did you do about it?**
Compare training and validation loss/mAP per epoch; a widening gap or rising val loss signals overfitting. Mitigations: transfer learning, augmentation, a fixed number of epochs with checkpoints (pick `best.pt` by validation fitness), weight decay.

**Q35. Which optimizer?**
Ultralytics' default `optimizer=auto` (SGD with momentum for long runs, AdamW for short ones), with weight decay. *(Check `args.yaml` of your run before quoting the exact one.)*

**Q36. How did you make training reproducible?**
Fixed seed 1337 for the split and training, pinned library versions, notebooks for every step, saved dataset YAML.

**Q37. How would you resume a crashed run?**
`YOLO('.../weights/last.pt').train(resume=True)` – we save checkpoints every 10 epochs.

**Q38. What hardware did you use and how long did it take?**
RTX 2060 6 GB with CUDA; roughly hours for 100 epochs. *(State the real duration from your run log.)*

**Q39. What does `imgsz=640` do and could higher help?**
Images are resized to 640 for the network. Higher (960/1280) keeps thin cracks visible and probably raises recall but costs memory and speed; a good future experiment.

### E. Evaluation metrics

**Q40. Define precision, recall, F1.**
Precision = TP/(TP+FP) – how many detections were right. Recall = TP/(TP+FN) – how many real damages were found. F1 = harmonic mean 2PR/(P+R). Ours: P 0.584, R 0.523, F1 0.552.

**Q41. What is IoU?**
Overlap area divided by union area of the predicted and true boxes; used to decide whether a detection counts as a true positive.

**Q42. What is AP, mAP, mAP@0.5 and mAP@0.5:0.95?**
AP = area under one class's precision-recall curve; mAP = mean over classes. @0.5 requires IoU ≥ 0.5 (lenient); @0.5:0.95 averages over IoU 0.5–0.95 (strict). Ours: 0.547 and 0.254.

**Q43. Why is mAP@0.5:0.95 so much lower than mAP@0.5?**
Detections are found but boxes aren't tight enough for high IoU thresholds. Crack extents are ambiguous and thin boxes are very sensitive to a few pixels' error.

**Q44. Why not use accuracy?**
Accuracy needs true negatives, which aren't defined in detection (infinite empty boxes) and would be dominated by background. Precision/recall/mAP evaluate the detections themselves.

**Q45. What does the confusion matrix tell you?**
It shows damage-type mix-ups (very few) and the main error modes: missed damage (predicted as background, ~190–253 per class) and false alarms on background (~133–224 per class). So the model rarely confuses classes; its main weakness is sensitivity to faint/thin damage.

**Q46. What does the PR curve show?**
Precision vs recall as the confidence threshold varies; precision stays above ~0.8 until recall ~0.3 then drops. It justifies the threshold slider.

**Q47. Which class performs best/worst and why?**
Best alligator (mAP@0.5 0.709): big, textured, most frequent. Worst transverse (0.454): thin, short, rare, confusable with joints/shadows.

**Q48. How do you choose the confidence threshold?**
It's a precision–recall trade-off. For screening (don't miss anything) use a low τ such as 0.25; for few false alarms use a higher τ such as 0.5. Ideally pick τ that maximises F1 on validation; the app exposes it as a slider (default 0.25 image / 0.5 video).

**Q49. What is NMS and its IoU threshold?**
Non-maximum suppression keeps the best-scoring box among heavily overlapping ones and removes the rest; Ultralytics' prediction default IoU threshold is 0.7.

**Q50. Is 54.7 % mAP good?**
It is a reasonable baseline for a noisy, thin-object problem but not production grade. Honest framing: strong on alligator cracks, weak on thin cracks; usable as a screening aid with a human in the loop.

**Q51. How would you improve the metrics?**
More data (other countries), larger model, higher resolution, crack-friendly augmentation, class re-balancing, tiling for small cracks, ensembling/TTA, cleaner labels, longer training, hyper-parameter tuning.

**Q52. How would you evaluate it properly in the real world?**
Collect a new set of roads/devices never used in training, evaluate on the whole pipeline (video, motion blur, night), measure per-class P/R and per-km detection counts against an engineer's survey.

### F. Image and video analysis

**Q53. Walk me through what happens when I upload an image.**
Read with PIL → fix EXIF rotation → convert to RGB → resize to 640×640 → convert to BGR → `YOLO.predict(conf=τ)` → draw boxes → resize back to original size → show original vs prediction, a detection table and a download button.

**Q54. Why EXIF handling?**
Phone photos store orientation in metadata; without `exif_transpose` the model receives a sideways image and detects poorly.

**Q55. Why convert to RGB?**
Uploads may be grayscale, palette or RGBA (PNG with transparency); the model needs exactly 3 channels.

**Q56. Why RGB↔BGR conversions?**
PIL/Streamlit use RGB; OpenCV/Ultralytics treat numpy arrays as BGR. Mismatched order swaps channels and subtly degrades predictions and colours.

**Q57. Why resize to 640×640 and what's the downside?**
It matches the training size. Downside: it ignores aspect ratio (no letterbox), slightly distorting wide images; letterboxing would be more faithful.

**Q58. How is video processed?**
Frame by frame with OpenCV: read, resize, predict, draw, resize back, write to an `.mp4` with `VideoWriter` at the original FPS; progress bar and live preview; then download.

**Q59. Can it run in real time?**
On the RTX 2060, ~13 ms/frame ≈ 75 FPS, so yes in principle. On the free cloud CPU ~160 ms/frame (≈ 6 FPS) – not real time; that's why the Realtime page is "coming soon".

**Q60. Why does the video page default to 0.5 while the image page uses 0.25?**
The image page's default was deliberately lowered to 0.25 after we observed that confidence on thin cracks is modest; the video page still carries the inherited 0.5. A stricter value does reduce flickering false positives across repeated frames, but the honest answer is that the two defaults should be aligned or chosen from the validation F1-vs-confidence curve. Both are user-adjustable.

**Q61. Does video detection track objects?**
No – each frame is independent, so a pothole is re-detected every frame and cannot be counted uniquely. Adding a tracker (ByteTrack/BoT-SORT) would enable counting and temporal smoothing.

**Q62. Why `mp4v` codec?**
pip OpenCV lacks H.264 due to licensing; `mp4v` always works, but browsers may not play it inline, so we provide a download.

**Q63. What happens with multiple users at once?**
Streamlit runs a script per session, but the video page uses fixed temp filenames, so simultaneous uploads can collide. Fix: unique temp names (e.g. `tempfile.NamedTemporaryFile`).

**Q64. How is the model loaded and cached?**
Once per session using `st.session_state`; the weights are downloaded if missing. A better approach is `@st.cache_resource` to share one copy across sessions.

**Q65. What would be different for a webcam/real-time feature?**
`streamlit-webrtc` for capture, a TURN server for NAT traversal on cloud, a lighter or GPU-served model, frame skipping and tracking.

### G. Deployment and engineering

**Q66. How did you deploy and what broke?**
Pushed to GitHub, connected the repo on Streamlit Community Cloud with `Home.py`. Fixes: `libgl1-mesa-glx` removed on Debian trixie → `libgl1`; `av==10.0.0` has no 3.11 wheel → `av==12.3.0`; CPU-only PyTorch wheel index for a lighter install.

**Q67. Why `opencv-python-headless`?**
Servers have no display; headless omits GUI libraries (still needs `libgl1`/`libglib2.0-0`).

**Q68. What is `packages.txt` vs `requirements.txt`?**
`requirements.txt` = Python (pip) packages; `packages.txt` = system (apt) packages on Streamlit Cloud.

**Q69. Why was the app throttled and what are the alternatives?**
Free tier limits CPU for heavy apps. Alternatives: Hugging Face Spaces (more RAM), Docker on a VM or Cloud Run, or a GPU host. Also reduce load: frame skipping, smaller input, cap video length.

**Q70. How would you scale it to many users?**
Separate the model into an inference API (FastAPI/Triton), use a GPU, queue video jobs (Celery/Redis), store results in object storage, add auth and rate limits.

**Q71. How would you run it on a phone or edge device?**
Export to ONNX / TFLite / CoreML / TensorRT (`model.export(...)`), possibly quantise to INT8, or use YOLOv8n for speed.

**Q72. How do you version models/data?**
Weights in the repo (or Git LFS / releases), fixed dataset YAML, seeds and pinned versions; for larger projects DVC/MLflow/W&B.

**Q73. What security/privacy issues exist?**
Uploaded images may contain faces/number plates; handle with consent, don't store beyond the session, blur plates, and validate/limit upload size and type (limit currently 1 GB).

**Q74. Licence considerations?**
Ultralytics YOLOv8 is AGPL-3.0 (copyleft – distributing/serving modified versions has obligations; commercial licence available); the dataset has its own research terms; the upstream project is credited.

### H. Tricky and "gotcha" questions

**Q75. Your mAP is only 0.547 – isn't that poor?**
For thin, low-contrast, noisily labelled objects it's an acceptable baseline; the best class reaches 0.709. mAP@0.5:0.95 is deliberately strict. We present it as a screening tool and list concrete ways to improve.

**Q76. Isn't your validation set also what you tuned on? Is that cheating?**
We did no heavy hyper-parameter search against it, but with no public test set the validation split is also the reporting set, so the numbers can be mildly optimistic. We say so; a held-out, road-disjoint test set is the right fix.

**Q77. Did you train the model yourselves?**
*(Answer truthfully, consistently with your team.)* If the baseline weights were used: "We used the published YOLOv8s road-damage weights from the upstream project as our baseline, re-ran the evaluation on the India validation split, and the training protocol is documented and reproducible in our notebooks." If you retrained: cite your run logs/`args.yaml`.

**Q78. The prep notebook says 3,594/399 but you validate on 1,294 images – why?**
The saved output in the prep notebook is from an earlier/upstream run; all reported results come from the evaluation run on the 1,294-image validation set. We can regenerate the split deterministically with seed 1337.

**Q79. Why are the image and video default thresholds different?**
See Q60 – the image default was tuned (0.25), the video default was left at the inherited 0.5. It is a known inconsistency, easy to fix, and both are adjustable by the user.

**Q80. Could the model detect a pothole in a photo of a water puddle/manhole/shadow?**
Possibly, those are known false-positive sources (we have 133–224 false alarms per class on background). More negative examples and hard-negative mining would reduce this.

**Q81. How does it handle night, rain, motion blur?**
Not specifically trained or evaluated for them; accuracy likely drops. Mitigation: augmentation (blur, brightness), more diverse data, image enhancement.

**Q82. What would you do with another month?**
Split-by-road test set; try YOLOv8m at 960 px; add class re-balancing and augmentation tuned to cracks; video tracking + counting; GPS tagging and a map dashboard; fix the video channel order and letterbox resize; Hugging Face/GPU deployment.

**Q83. What ethical concerns are there?**
Misdetections could misdirect repair funds or give false safety assurance; mitigate with a human in the loop and reporting confidence; privacy of captured images; fairness across regions with different road types.

**Q84. How do you know the app output is correct and not just pretty boxes?**
Quantitatively via validation metrics, qualitatively by comparing predictions with ground truth on validation batches (`figures/val_batch2_labels.jpg` vs `val_batch2_pred.jpg`), and by testing the app on external images.

**Q85. Explain the difference between bias and variance in this project.**
High bias (underfitting) would show low scores on both train and val – e.g. a too-small model missing thin cracks. High variance (overfitting) would show great train but poor val scores – risk with limited data. Transfer learning, augmentation and a mid-sized model balance the two.

---

## 16. Team, credits, licences

**Team (B.Tech CSE, NIT Delhi):** Devam Kharbanda (241210037) · Gufran Alam (241210042) · Harshit Jhamb (241210047) · Abhiram Kavitharkika (241210058) · Krish Karandikar (241210065) – supervised by **Dr. Gautam Kumar**.

**Credits and licences**
- Builds on the open-source [RoadDamageDetection](https://github.com/oracl4/RoadDamageDetection) project by **oracl4**, whose training protocol, baseline YOLOv8s weights and Streamlit app structure we studied, re-ran, evaluated and extended. Full credit to the original author; the repository history preserves their commits.
- Dataset: **RDD2022 / CRDDC2022** – Arya, Maeda, Ghosh, Toshniwal, Sekimoto, *"RDD2022: A multi-national image dataset for automatic road damage detection"*, arXiv:2209.08538.
- Model framework: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) (AGPL-3.0).
- Web framework: [Streamlit](https://streamlit.io/). Computer vision: [OpenCV](https://opencv.org/). Deep learning: [PyTorch](https://pytorch.org/).
- Further reading: Redmon et al., *You Only Look Once* (CVPR 2016); Lin et al., *Microsoft COCO* (ECCV 2014).

Full write-up with figures: [`04_ML_Project_Report.docx`](04_ML_Project_Report.docx).
