# Gender Classification CNN – CelebA

A binary gender classifier (Female / Male) built with a custom CNN in TensorFlow/Keras, trained on the [CelebA Gender Recognition 200K Images](https://www.kaggle.com/datasets/ashishjangra27/gender-recognition-200k-images-celeba) dataset. Includes a FastAPI web server with a drag-and-drop UI for real-time inference.

[![Kaggle Notebook](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/majdkhuzam/gender-classification-cnn-celeba)
[![Hugging Face Space](https://img.shields.io/badge/🤗%20Hugging%20Face-Space-blue)](https://huggingface.co/spaces/MajdKhuzam/gender-classification-celeba)
[![Live Demo](https://img.shields.io/badge/🚀%20Live-Demo-brightgreen)](https://huggingface.co/spaces/MajdKhuzam/gender-classification-celeba)

---

## Project Structure

```
gender-classification-cnn-celeba/
├── data_loader.py            # Dataset loading & preprocessing
├── train.py                  # CNN architecture + training
├── evaluate.py               # Evaluation & metrics
├── visualize.py              # Prediction visualisation (3×3 grid)
├── main.py                   # FastAPI web server (/predict, /health, static UI)
├── Dockerfile                # Container image definition
├── static/
│   ├── index.html            # Drag-and-drop web interface
│   ├── main.js               # Front-end logic
│   └── styles.css            # Styles
├── output/
│   └── gender_classification_model.keras
├── requirements.txt
└── .gitignore
```

---

## Model Architecture

| Layer | Details |
|---|---|
| Conv Block 1 | Conv2D(32) → BatchNorm → MaxPool |
| Conv Block 2 | Conv2D(64) → BatchNorm → MaxPool |
| Conv Block 3 | Conv2D(64) → BatchNorm → MaxPool |
| Dense Head | Dense(64, relu) → Dropout(0.2) → Dense(1, sigmoid) |

- **Input:** 128 × 128 × 3 (RGB)
- **Output:** Single sigmoid unit — probability of *Male* class
- **Loss:** Binary cross-entropy
- **Optimizer:** Adam

---

## Setup

```bash
git clone https://github.com/MajdKhuzam/gender-classification-celeba
cd gender-classification-celeba
pip install -r requirements.txt
```

---
## Pre-trained Model

A trained model is available on Hugging Face — no dataset or training required:

[Download gender_classification_model.keras](https://huggingface.co/spaces/MajdKhuzam/gender-classification-celeba/blob/main/output/gender_classification_model.keras)

Place it at `output/gender_classification_model.keras` to use with `evaluate.py` or `main.py`.

---

## Dataset

Download from Kaggle:
[ashishjangra27/gender-recognition-200k-images-celeba](https://www.kaggle.com/datasets/ashishjangra27/gender-recognition-200k-images-celeba)

Extract into the project root so the path is:

```
Data/gender-recognition-200k-images-celeba/Dataset/
```

Expected directory layout:

```
Dataset/
├── Train/
│   ├── Female/
│   └── Male/
├── Test/
│   ├── Female/
│   └── Male/
└── Valid/
    ├── Female/
    └── Male/
```

---

## Usage

### 1. Train

```bash
python train.py
```

Saves the trained model to `output/gender_classification_model.keras`.

### 2. Evaluate

```bash
python evaluate.py
```

Prints test loss, accuracy, a full classification report, and a confusion matrix.

### 3. Visualise predictions

```bash
python visualize.py
```

Displays a 3×3 grid of random test images. Green title = correct prediction, red = incorrect.


### 4. Web server (inference API + UI)

```bash
python main.py
```

Then open http://localhost:7860 in your browser.

**API endpoints:**

| Endpoint | Method | Description |
|---|---|---|
| `/predict` | POST | Upload an image file; returns `{ label, confidence, raw_probability }` |
| `/health` | GET | Server health check (`{ status, model_loaded }`) |

The root path serves a cyberpunk-themed web UI with drag-and-drop image upload and live classification results.

---

## Evaluation Results

Evaluated on **20,001 test images** from the CelebA dataset.

| Metric | Value |
|---|---|
| Test Loss | 0.0783 |
| Test Accuracy | **98.04%** |

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Female (0) | 0.98 | 0.98 | 0.98 | 11,542 |
| Male (1) | 0.98 | 0.97 | 0.98 | 8,459 |
| **Accuracy** | | | **0.98** | **20,001** |
| Macro avg | 0.98 | 0.98 | 0.98 | 20,001 |
| Weighted avg | 0.98 | 0.98 | 0.98 | 20,001 |

### Confusion Matrix

|  | Predicted Female | Predicted Male |
|---|---|---|
| **Actual Female** | 11,367 | 175 |
| **Actual Male** | 218 | 8,241 |

- **False Positives** (Female predicted as Male): 175
- **False Negatives** (Male predicted as Female): 218

---

## Requirements

| Package | Version |
|---|---|
| tensorflow | 2.19.0 |
| numpy | 2.0.2 |
| opencv-python-headless | 4.12.0.88 |
| scikit-learn | 1.6.1 |
| matplotlib | 3.10.0 |
| fastapi | 0.136.3 |
| uvicorn | 0.48.0 |
| python-multipart | 0.0.29 |
| gunicorn | 23.0.0 |


