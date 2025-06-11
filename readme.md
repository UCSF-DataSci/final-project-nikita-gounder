# Final Project Report: CNN Classification of Blood Cell Types
Nikita Gounder
DS 224
6/11/25

## Introduction
White blood cell classification is an important task for clinical diagnostics, as abnormal levels or appearances of specific cell types can indicate infections, immune disorders, or blood cancers like leukemia. Accurate identification of WBC types from microscopic images supports timely and effective treatment decisions. However, existing models often suffer from issues like overfitting, class imbalance, or limited generalization due to small or imbalanced datasets. In this project, I use the BloodMNIST dataset from MedMNIST, which contains labeled RGB images of various WBC types, to develop a convolutional neural network (CNN) for automated classification. The aim is to explore how well a deep learning pipeline using TensorFlow/Keras performs on this task, while evaluating performance using standard metrics such as accuracy, precision, recall, F1-score, and confusion matrices. [1]

## Datset
The BloodMNIST from MedMNIST is a dataset of 26,27 of microscop images of individual normal cells, captured from individuals without infection, hematologic or oncologic disease and free of any pharmacologic treatment at the moment of blood collection. It contains a total of 17,092 images and is organized into 8 classes ('0': 'basophil', '1': 'eosinophil', '2': 'erythroblast', '3': 'immature granulocytes(myelocytes, metamyelocytes and promyelocytes)', '4': 'lymphocyte', '5': 'monocyte', '6': 'neutrophil', '7': 'platelet'). The dataset is already split with a ratio of 7:1:2 into training, validation and test sets. The source images with resolution 3 × 360 × 363 pixels are center-cropped into 3 × 200 × 200, and then resized into 3 × 28 × 28. [2]


## Methods




## Results




## Discussion

Tried to add BatcNorm but -->

Possible Reasons BatchNorm Reduces Accuracy
1. Your model is already simple and well-behaved

BatchNorm helps with deep or unstable networks. If your CNN is small (e.g., 2–3 conv layers), it might not need it.
Adding BN can introduce unnecessary regularization and slightly slow learning.
2. Wrong placement

BN is most effective when used after the layer's linear transformation and before activation (e.g., Conv → BN → ReLU).
If you accidentally placed it before the conv layer or after ReLU, it may hurt performance.
3. Small batch size

BatchNorm estimates the mean and variance within each batch.
If your batch size is small (e.g., <16), those estimates are noisy, which can make learning unstable or suboptimal.
4. Mismatch between training and inference modes

In training, BN uses batch stats; in inference, it uses running averages.
If model.eval() isn’t called properly during evaluation, test-time behavior may still rely on batch stats, which are unreliable.
5. Learning rate too high after adding BN

BN lets you use higher learning rates, but if you don’t tune it accordingly, performance may degrade.


## References
1. https://pmc.ncbi.nlm.nih.gov/articles/PMC9691098/
2. https://www.nature.com/articles/s41597-022-01721-8
3. https://www.sciencedirect.com/science/article/abs/pii/S1746809421007539
4. 