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

Conclusion & Key Findings
While several models showed strong performance, Inception proved to be the most effective model for our dataset, effectively distinguishing between fungal and bacterial patterns.

Metric	Result

Best Model :	 InceptionV3

Highest Accuracy :	97.00%

Best Split Ratio :	70:30

Inception was selected for the final deployment due to its excellent balance of feature extraction capabilities and overall classification accuracy on the Betel leaf dataset.




