# Technical Report: Coffee Bean Classifier


---

## 1. Software Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Framework | TensorFlow / Keras | Model training & inference |
| Model Architecture | MobileNetV2 (Transfer Learning) | Image classification |
| Computer Vision | OpenCV | Webcam capture & preprocessing |
| Numerical Computing | NumPy | Array manipulation & normalization |
| Visualization | Matplotlib | Accuracy plots |

**Environment:** Python 3.14 + pip dependencies

---

## 2. Dataset & Preprocessing

A curated dataset of **120+ images** was used, categorized into:
- **GOOD** — Roasted coffee beans
- **BAD** — Green / defective beans

### Bug #01 Fix: Lighting Variations

To resolve low-light classification failures, data augmentation was implemented using Keras `ImageDataGenerator`:

| Augmentation | Parameter | Purpose |
|--------------|-----------|---------|
| Random rotations | ±40° | Orientation invariance |
| Brightness shifts | 0.7 – 1.3 | Simulate factory lighting |
| Horizontal flips | Enabled | Mirror symmetry |

**Outcome:** Model works accurately under varied lighting conditions.

---

## 3. Model Architecture & Selection

### Primary Model: MobileNetV2 (Transfer Learning)

| Layer | Description |
|-------|-------------|
| Base Model | MobileNetV2 pre-trained on ImageNet (1.4M images) |
| Top Layers | Removed (frozen) |
| Pooling | Global Average Pooling |
| Output | Single Dense + Sigmoid (Binary: GOOD/BAD) |

### Benchmark Model: Simple CNN (from scratch)

| Model | Validation Accuracy | Generalization |
|-------|---------------------|----------------|
| Simple CNN | 99% | High on training data |
| MobileNetV2 | 99% | Better real-world reliability |

**Decision:** MobileNetV2 selected for deployment — resistant to overfitting (Bug #02) and better real-world performance.

---

## 4. Folder Architecture


coffee-project/
│
├── data/
│   ├── GOOD/
│   └── BAD/
│
├── models/
│   ├── mobilenetv2_model.h5
│   └── cnn_model.h5
│
├── src/
│   ├── inference.py
│   ├── train.py
│   └── test_webcam.py
│
├── outputs/
│   ├── accuracy_plot.png
│  
│
├── requirements.txt
└── README.md
```

---
## 5. Key Technical Challenges

| Bug | Issue | Solution |
|-----|-------|----------|
| #01 | Model fails in low light | Brightness augmentation (0.7–1.3 range) + rotations (±40°) |
| #02 | Overfitting | MobileNetV2 transfer learning (pre-trained on ImageNet) |

---

## 6. Performance Results

| Model | Validation Accuracy | Generalization |
|-------|---------------------|----------------|
| Simple CNN | 99% | High on training data |
| MobileNetV2 | 99% | Superior real-world reliability |

> ✅ Target: ≥ 80% | Achieved: **99%**

---

## 7. Conclusion & Recommendations

The MobileNetV2 classifier provides an optimal balance of **speed, low latency, and high-precision accuracy** for the current one-by-one sorting gate prototype.

**Future Recommendation:** Transition to YOLO (Object Detection) if requirements shift toward scanning entire bags of beans simultaneously.

---

**Report by:**  Nole Mohammed   
**Contributors:** Samuel , Efrata 
**Date:** April,26,2026