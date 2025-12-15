# Betel Leaf Image Dataset From Bangladesh

Description: The Betel Leaf Image dataset is a curated collection of original and augmented images of betel leaves, categorized into healthy and diseased conditions. The dataset primarily focuses on the cultivation context of Bangladesh and covers common leaf impairments such as Bacterial leaf disease, Dried leaf, Fungal Brown spot disease, Healthy leaf.

Categories: There are 4 classes in this dataset

Bacterial leaf disease: 250 images 

Dried leaf: 250 images 

Fungal Brown spot disease: 250 images 

Healthy leaf: 250 images 

Total images 1000 

Dataset link : https://data.mendeley.com/datasets/g7fpgj57wc/2

Performance Overview : We evaluated several state-of-the-art pre-trained deep learning models to identify the optimal architecture for this specific task of Betel leaf image dataset from Bangladesh.

Tested Architectures 

The following models were benchmarked:

Alexnet

DenseNet121

EfficientNet

InceptionV3 (Winner)

MobileNetV2

VGG16

ResNet50

Xception

Methodology : To ensure robustness, each model was tested across a spectrum of data distributions. We utilized distinct Train-Test splits to validate generalizability, with a primary focus on the 70:30 distribution.

Conclusion & Key Findings:

While several models showed strong performance, Inception proved to be the most effective model for our dataset, effectively distinguishing between fungal and bacterial patterns.

Best Model :	 InceptionV3

Highest Accuracy :	97.00%

Best Split Ratio :	70:30

Inception was selected for the final deployment due to its excellent balance of feature extraction capabilities and overall classification accuracy on the Betel leaf dataset.

Repository Structure:
Betel-Leaf-Image-Dataset-from-Bangladesh/
│
├── Code/                                       # Source code and Jupyter Notebooks
│   | For Transfer Learning Models
│   ├───inceptionv3.ipynb                   # 🏆 Best Performing Model (97%)
│   ├── resnet50.ipynb                      # ResNet50 Implementation
│   ├── mobilenetv2.ipynb                   # MobileNetV2 Implementation
│   ├── densenet121.ipynb                   # DenseNet121 Implementation
│   ├── vgg16.ipynb                         # VGG16 Implementation
│   ├── xception.ipynb                      # Xception Implementation
│   ├── alexnet.ipynb                       # AlexNet Implementation
│   ├── efficientnetb0.ipynb                # EfficientNetB0 Implementation
│   ├── efficientnetb0-v2.ipynb             # Improved EfficientNet variant
│   │
│   | For Custom & Attention Models
│   ├── custom-cnn.ipynb                    # Task 3: Baseline Custom CNN
│   ├── BLCNN.ipynb                         # Task 3: Betel Leaf Specific CNN
│   ├── cbam-cnn.ipynb                      # Task 4: CNN with CBAM Attention
│   ├── SE-ResNet.ipynb                     # Task 4: ResNet with Squeeze-and-Excitation
│   ├── SE Custom CNN.ipynb                 # Task 4: Custom CNN with SE blocks
│   ├── SE Custom CNN [Updated Plot].ipynb   # Updated plotting for SE model
│   ├── Custom Hybrid CNN-Swin Transformer.ipynb # Task 4: Hybrid Transformer Model
│   │
│   | For Analysis & Utilities
│   ├── EDA-betel-leaf-image-dataset-from-Bangladesh.ipynb # Exploratory Data Analysis for Task 1
│   ├── Grad-CAM of Custom CNN.ipynb        # Model Interpretability/Visualization for Task 5
│   ├── generalizability-testing-inceptionv3.ipynb # Robustness testing for Task 5
│   ├── Paired-t-test-for-task-2.ipynb      # Statistical significance testing for Task 2
│   ├── Task 2.md                           # Specific markdown report for Task 2
│   │
│   └── ├── images/                                     # Result visualizations for Task 2.md
│       ├── inceptionv3_results.png
│       ├── densenet_finetuned_results.png
│       └── inceptionv3_confusion_matrix.png           
│
├── Report/                                    
│   ├── Task1/                                  
│   ├── Task2/                                 
│   ├── Task3/                                  
│   ├── Task4/                                  
│   └── Report                                  
│
└── README.md                                   # Project Overview (This file)
