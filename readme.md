# Final Project Report: CNN Classification of Blood Cell Types
Nikita Gounder
DS 224
6/11/25

## Introduction
Classifying blood cells, especially white blood cells (WBC), is crucial in clinical diagnostics. Abnormal levels or appearances can indicate infections, immune disorders, or blood cancers. Accurate identification of cell types from microscopic images is essential for timely and effective treatment decisions. Existing models, however,r often suffer from overfitting, class imbalance, or limited generalization due to small or imbalanced datasets. In this project, I use the BloodMNIST dataset from MedMNIST, which contains labeled RGB images of various blood cell types, to develop a convolutional neural network (CNN) for automated classification. The aim is to explore how well a deep learning pipeline using TensorFlow/Keras performs on this task while evaluating performance using standard metrics such as accuracy, precision, recall, F1-score, and confusion matrices.[1]


## Datset
The BloodMNIST dataset from MedMNIST is of microscope images of individual normal cells from individuals without infection, hematologic or oncologic disease and free of any pharmacologic treatment at the moment of blood collection. It contains a total of 17,092 images and is organized into 8 classes ('0': 'basophil', '1': 'eosinophil', '2': 'erythroblast', '3': 'immature granulocytes(myelocytes, metamyelocytes and promyelocytes)', '4': 'lymphocyte', '5': 'monocyte', '6': 'neutrophil', '7': 'platelet'). The dataset is already split into training, validation, and test sets with a ratio of 7:1:2. The source images are center-cropped and resized to 3 × 28 × 28. [2]


## Methods
Traditional machine learning methods, such as Support Vector Machines (SVMs) and Bayesian classifiers, have been used for WBC classification due to their lower data requirements and simpler implementation. However, these approaches typically rely on extensive preprocessing, including image segmentation and hand-crafted feature extraction, which can introduce bias and require domain expertise. In contrast, convolutional neural networks (CNNs) can learn relevant features directly from raw pixel data, reducing the need for manual feature engineering. Previous studies have shown CNNs often achieve comparable or superior performance to these methods when trained on sufficiently large, labeled datasets. For this project, I used a CNN implemented in TensorFlow/Keras to leverage its ability to automatically learn spatial features from blood cell images in the BloodMNIST dataset. [3]

For this project, I chose the convolutional neural network using Keras (via TensorFlow) instead of PyTorch. While both frameworks are widely used and powerful, Keras is more straightforward and allows for a more streamlined workflow. In contrast, PyTorch is more flexible and controlled but requires more setup. For the straightforward nature of this classification task and my goal to build a working deep learning pipeline efficiently, Keras seemed to be the best choice.

First was the preprocessing steps to prepare the BloodMNIST dataset for use with Keras. The dataset was downloaded from MedMNIST, with separate splits for training, validation, and testing. I extracted the image arrays and labels from each split, then normalized the image pixel values to the range [0, 1] by dividing by 255 and converting them to float32 for compatibility with TensorFlow operations. Since the classification task involves predicting one of eight blood cell types, I applied one-hot encoding to the categorical labels using the to_categorical function. This converted the label arrays into binary matrices for use with the categorical_crossentropy loss function. At this stage, all input data had the shape (11959, 28, 28, 3) representing the number of images, image height, width, and RGB channels. 

The CNN architecture begins with an input layer for the 28x28 RGB image format of the dataset. I added two convolutional layers with ReLU activation functions followed by max pooling layers to reduce dimensionality and computation. A flattening layer was used to convert the feature maps into a one-dimensional vector, which was passed through dense layers to learn complex feature representations. The final output layer uses a softmax activation function to perform multi-class classification. I compiled the model using the Adam optimizer and categorical cross-entropy loss, with accuracy as the main evaluation metric. 

During training, I applied early stopping to prevent overfitting and monitored performance on a validation set. I trained the model for 15 epochs to balance training time with performance. After training, I evaluated the model using metrics such as precision, recall, and F1-score and visualized results using a confusion matrix to assess classification performance across classes.


## Results and Discussion
The CNN model was trained on the BloodMNIST dataset for up to 15 epochs. Early stopping with restore_best_weights=True was applied based on validation loss, and the model achieved its best performance at epoch 13, with a validation accuracy of 92.29% and validation loss of 0.2189. While training continued until epoch 15, early stopping made sure the final model was set to epoch 13.

I tested Batch normalization, but it was ultimately removed because it resulted in a drop in model accuracy during validation. Throughout the training, both training and validation accuracy improved steadily. The training accuracy reached 92.17% by epoch 15, while the validation accuracy plateaued around 92%, indicating strong generalization without signs of overfitting. 

Evaluation of the test set produced the metrics seen in the code. Notably, test accuracy was 91.61%, and precision was 92.21%. The metrics suggest that the model performed consistently across all classes, with minimal class imbalance effects due to normalization. The confusion matrix reveals per-class performance. For example, Class 1 (eosinophils) had high precision with 616 correct predictions and few misclassifications. In contrast, Class 3 (immature granulocytes) had more confusion, with misclassifications spread across other classes, such as 0 and 6. This may show overlapping visual features or class imbalance in the original data.

Overall, the model had strong multi-class classification performance. These results show that a lightweight CNN can effectively classify blood cell images and is promising for clinical settings where fast, accurate, and computationally efficient models are needed for diagnostic assistance.


## References
1. https://pmc.ncbi.nlm.nih.gov/articles/PMC9691098/
2. https://www.nature.com/articles/s41597-022-01721-8
3. https://www.sciencedirect.com/science/article/abs/pii/S1746809421007539