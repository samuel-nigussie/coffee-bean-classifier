# coffee-bean-classifier
 
> **Applied AI & Interdisciplinary Integration**  
> *April 2026*

---

##  Overview

This project builds a real-time **coffee bean quality classifier** using a webcam and a deep learning model.  
The system classifies each bean frame as **GOOD** or **BAD** and displays the result with a green/red indicator — simulating an automated optical sorter for coffee production lines.

This prototype was developed as part of an internal rapid prototyping sprint, focusing on:
- Transfer learning with MobileNetV2
- Real-time webcam inference using OpenCV
- Performance comparison with a simple CNN
- Professional GitHub documentation & system architecture

---

## Team Members & Roles

| Member         | Role                                      | Key Deliverable                               |
|----------------|-------------------------------------------|-----------------------------------------------|
| **Samuel**     | Systems Architect & Integration Lead      | Webcam verification, inference bridge, **UI optimization**  |
| **Nole**       | Data Science & Performance                | Trained models (MobileNetV2 + CNN), validation ≥ 80%, **technical report** |
| **Efrata**     | AI Integration & Real-Time Systems        | Inference pipeline, FPS tuning, integration support |

---

##  Models

Two models were trained and compared:

1. **MobileNetV2 (Transfer Learning)**  
   - Input size: 224×224  
   - Normalized pixel values  
   - Frozen base layers + custom dense head  
   - *Primary model*

2. **Simple CNN (from scratch)**  
   - Built for performance comparison  
   - Fewer parameters, faster training

>  Primary model validation accuracy: **≥ 80%**

--
