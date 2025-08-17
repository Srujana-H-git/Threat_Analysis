# UAV Threat Detection using CNN

This project implements a **Convolutional Neural Network (CNN)** from scratch for detecting threats in UAV-captured images.  
The model is trained on custom datasets organized into `train`, `val`, and `test` folders, each containing two classes:  
- `threat`  
- `non_threat`

---

## 🚀 Features
- End-to-end **CNN-based image classification** model.
- Trains on custom UAV datasets with two classes.
- Includes data preprocessing, augmentation, training, validation, and testing.
- Evaluation with accuracy, classification report, and confusion matrix.
- Threat analysis using predicted probabilities.

---

## 📊 Evaluation

After training, the script will:
Evaluate the model on the test set.
Display:
Accuracy
Classification Report (Precision, Recall, F1-score)
Confusion Matrix
Perform threat probability analysis to assess detection performance.

---

## 🛡️ Threat Analysis

The model outputs probability scores for each prediction.
High probability → More confidence in prediction.
Low probability → Indicates uncertainty.
This helps in prioritizing UAV alerts based on confidence levels.

---


## 📈 Future Improvements

Use transfer learning with ResNet, EfficientNet, or ViT for better accuracy.
Deploy model on Raspberry Pi for real-time UAV monitoring.
Integrate with LoRa/RF communication for field transmission.

---
