# 🧘 Yoga Pose Classification with Transfer Learning

A deep learning–based computer vision system that accurately classifies yoga poses from images using **transfer learning and fine-tuning**.

The model achieves **93.6% test accuracy** and demonstrates strong generalization across pose variations.

---

## Project Overview

This project builds an end-to-end pose classification pipeline that:

* classifies yoga poses from images
* distinguishes visually similar poses using posture geometry
* leverages transfer learning for high accuracy with moderate data
* evaluates performance using confusion analysis and class metrics

The system is designed to be robust to variations in lighting, perspective and pose execution.

---

## Supported Poses

* Downward Dog
* Goddess
* Planks
* Tree
* Warrior II

---

## Model Architecture

### Backbone

**MobileNetV2 (ImageNet pretrained)**
Chosen for its efficiency and strong performance in human pose tasks.

### Training Strategy

**Phase 1 — Feature Extraction**

* frozen backbone
* train custom classification head

**Phase 2 — Fine-Tuning**

* unfreeze top layers
* refine pose-specific features

### Classification Head

* Global Average Pooling
* Dense layers with L2 regularization
* Batch Normalization
* Dropout for generalization

---

## Data Augmentation

To improve robustness and pose invariance:

* rotation (±15°)
* vertical & horizontal shifts
* zoom & shear transformations
* horizontal flipping

Vertical shifts were especially helpful for learning **hip height differences** between poses like plank vs. downward dog.

---

## 📊 Results

### Final Performance

| Metric            | Score      |
| ----------------- | ---------- |
| **Test Accuracy** | **93.62%** |
| Macro F1          | 0.93       |
| Weighted F1       | 0.94       |

### Class Performance

| Pose         | Precision | Recall | F1   |
| ------------ | --------- | ------ | ---- |
| Downward Dog | 0.99      | 0.96   | 0.97 |
| Goddess      | 0.98      | 0.80   | 0.88 |
| Plank        | 0.96      | 0.97   | 0.96 |
| Tree         | 0.90      | 0.94   | 0.92 |
| Warrior II   | 0.87      | 0.98   | 0.92 |

---

## Confusion Analysis

The model performs strongly across poses.

Minor confusion occurs between **Goddess** and **Warrior II** due to similar lower-body stance and hip angles.

---

## ⚙️ Training Pipeline

### Phase 1

* Learning rate: `3e-4`
* Frozen backbone
* Rapid feature alignment

### Phase 2

* Learning rate: `1e-4 → 2.5e-5`
* Fine-tuning last layers
* Improved pose geometry discrimination

### Regularization & Stability

* Dropout (0.5 / 0.4 / 0.3)
* L2 weight decay
* ReduceLROnPlateau
* EarlyStopping with best weight restore

## 🛠 Tech Stack

* Python
* TensorFlow / Keras
* MobileNetV2
* OpenCV & PIL
* NumPy / Pandas
* Matplotlib / Seaborn
* Scikit-learn

---

## 📂 Project Structure

```
├── TRAIN/
│   ├── downdog/
│   ├── goddess/
│   ├── plank/
│   ├── tree/
│   └── warrior2/
│
├── TEST/
├── yoga_pose_final.h5
├── solution.ipynb
└── README.md
```
Dataset is available here: https://www.kaggle.com/datasets/niharika41298/yoga-poses-dataset

---

## ▶️ How to Run

### 1️⃣ Install dependencies

```bash
pip install tensorflow matplotlib seaborn scikit-learn pillow
```

### 2️⃣ Train model

Run the training notebook or script.

### 3️⃣ Evaluate

The script generates:

* accuracy & loss curves
* confusion matrix
* classification report
