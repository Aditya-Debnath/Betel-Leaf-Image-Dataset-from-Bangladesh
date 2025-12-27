
# Betel Leaf Disease Classification using Deep Learning

![Project Status](https://img.shields.io/badge/Status-Completed-success)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## 📌 Project Overview
Betel leaf (*Piper betle L.*) is a significant cash crop in South and Southeast Asia. However, it is highly susceptible to diseases like Bacterial Leaf Spot, Foot Rot, and Fungal Brown Spot, which cause substantial economic losses. Traditional manual diagnosis is labor-intensive and subjective.

This project presents a comprehensive comparative study of **12 deep learning architectures** to automate the detection of betel leaf diseases. We propose a novel **Custom Hybrid CNN-Swin Transformer** that achieves state-of-the-art performance by synergizing local feature extraction (CNN) with global context modeling (Transformers).

## 🎯 Objectives
*   **Evaluate Baselines:** Compare custom CNNs against standard Transfer Learning models (InceptionV3, ResNet, etc.).
*   **Enhance Performance:** Integrate attention mechanisms (CBAM, SE-Blocks) and Transformer architectures.
*   **Ensure Robustness:** Test models on both balanced and highly imbalanced datasets.
*   **Explainability:** Implement Explainable AI (XAI) using Grad-CAM to visualize decision boundaries.

## 📂 Dataset
The study utilizes two datasets derived from the "Betel Leaf Image Dataset from Bangladesh":
1.  **Dataset 1 (Balanced):** 1,000 images (250 per class) used for training and primary benchmarking.
2.  **Dataset 2 (Imbalanced):** A larger, real-world dataset used to test model robustness and generalization.

## Links of dataset
**Dataset 1 (Balanced)** link : https://data.mendeley.com/datasets/g7fpgj57wc/2
**Dataset 2 (Imbalanced)** link : https://data.mendeley.com/datasets/vpzkntzjty/1

**Classes:**
1.  Bacterial Leaf Disease
2.  Dried Leaf
3.  Fungal Brown Spot Disease
4.  Healthy Leaf

## 🏗️ Methodologies & Architectures
We implemented and evaluated three categories of models:

### 1. Transfer Learning Baselines
Pre-trained models on ImageNet, fine-tuned for this specific task:
*   *InceptionV3, ResNet50, MobileNetV2, DenseNet121, EfficientNet-B0, Xception, VGG16, AlexNet.*

### 2. Custom Implementations
*   **Custom Sequential CNN:** A standard baseline lightweight network.
*   **BLCNN (Betel Leaf CNN):** Domain-specific architecture with depth-wise separable convolutions.

### 3. Advanced Attention & Hybrid Models
*   **Hybrid CNN-Swin Transformer:** Fuses a CNN stem with Swin Transformer blocks (Window-based Multi-head Self-Attention).
*   **CBAM-CNN:** Integrates Convolutional Block Attention Modules (Spatial + Channel attention).
*   **SE-ResNet / SE-Custom:** Enhances architectures with Squeeze-and-Excitation blocks.

## 📊 Results
The models were evaluated based on Accuracy, F1-Score, and Training Time.

| Model Architecture | Test Accuracy | Observations |
|--------------------|---------------|--------------|
| **Hybrid CNN-Swin**| **98.0%**     | **Best Overall.** Excellent generalization on imbalanced data (93%). |
| CBAM-CNN           | 97.3%         | Strong focus on lesion textures due to spatial attention. |
| InceptionV3        | 96.7%         | Best standard Transfer Learning model. Fast and stable. |
| ResNet50           | 95.0%         | Reliable, but slightly heavier computation. |
| MobileNetV2        | 94.0%         | Good balance of speed and accuracy. |
| Custom CNN         | 90.0%         | Good baseline but struggles with subtle texture differences. |

## 🧠 Explainable AI (XAI)
We utilized **Grad-CAM** to verify that the models focus on the actual diseased regions of the leaf rather than background noise. The Hybrid Swin Transformer showed the most precise localization of necrotic spots and bacterial lesions.

## 🚀 Usage

### Prerequisites
```bash
pip install torch torchvision opencv-python matplotlib scikit-learn seaborn torchinfo
