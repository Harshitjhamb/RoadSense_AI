# RoadSense AI – AI-Based Road Damage and Pothole Detection

RoadSense AI detects and classifies road surface damage from images and videos using a YOLOv8 deep learning model trained on the India subset of the Crowdsensing-based Road Damage Detection Challenge 2022 (CRDDC2022 / RDD2022) dataset. It is a B.Tech (CSE), 5th semester Machine Learning project (CSBB 311) at the National Institute of Technology Delhi.

Detected damage types:
- Longitudinal Crack
- Transverse Crack
- Alligator Crack
- Pothole

## Features
- **Image Detection** – upload an image, adjust the confidence threshold, view annotated output with a table of detections, and download the result.
- **Video Detection** – upload a road video and download the annotated video.
- **Realtime Detection** – coming soon, after further lab evaluation.

## Our Contributions
- Redesigned Home page and multi-page Streamlit app for the project.
- Fixes to the image pipeline: RGB normalisation, EXIF rotation handling, consistent RGB/BGR conversion, extra image formats, default confidence of 0.25, and a detection summary table.
- Restricted training and evaluation to the India subset and re-ran the evaluation notebook (see results below).
- Realtime page replaced with a "coming soon" notice pending lab evaluation.
- Full project report in `docs/ML Project Report.docx`.

## Results (validation set: 1,294 images, 2,272 instances)

| Class | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|
| Longitudinal Crack | 0.573 | 0.471 | 0.501 | 0.235 |
| Transverse Crack | 0.522 | 0.433 | 0.454 | 0.177 |
| Alligator Crack | 0.659 | 0.684 | 0.709 | 0.373 |
| Pothole | 0.584 | 0.503 | 0.524 | 0.229 |
| **All** | **0.584** | **0.523** | **0.547** | **0.254** |

Inference: about 13 ms per image on an NVIDIA RTX 2060.

<p align="center">
    <img src='resource/PR_curve.png' width='80%'>
    <img src='resource/confusion_matrix.png' width='80%'>
    <img src='resource/val_batch2_pred.jpg' width='100%'>
</p>

## Run Locally

```bash
conda create -n roadsense python=3.8
conda activate roadsense
# Install PyTorch (CUDA optional): https://pytorch.org/get-started/locally/
pip install -r requirements.txt
streamlit run Home.py
```

## Training

1. Download RDD2022 from the [RoadDamageDetector](https://github.com/sekilab/RoadDamageDetector) page and keep only the `India` folder.
2. Run `training/0_PrepareDatasetYOLOv8.ipynb` – converts Pascal VOC to YOLO format, keeps only the four classes, retains a limited share of background images, and makes a 90/10 train/validation split (seed 1337).
3. Run `training/1_TrainingYOLOv8.ipynb` – YOLOv8s, 100 epochs, batch 32, 640 px, cosine LR, mosaic off.
4. Run `training/2_EvaluationTesting.ipynb` – precision, recall, mAP and confusion matrix.

## Team
Devam Kharbanda, Gufran Alam, Harshit Jhamb, Abhiram Kavitharkika, Krish Karandikar – supervised by Dr. Gautam Kumar, NIT Delhi.

## Acknowledgements and Licenses
This project builds on the open-source [RoadDamageDetection](https://github.com/oracl4/RoadDamageDetection) project by oracl4, whose training protocol, baseline YOLOv8s weights and Streamlit app structure we studied, re-ran, evaluated and extended. Full credit to the original author; the repository history preserves their commits.

- Dataset: RDD2022 / CRDDC2022 (Arya et al.).
- Model framework: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) (AGPL-3.0).
- Web framework: [Streamlit](https://streamlit.io/).
# RoadSense_AI
