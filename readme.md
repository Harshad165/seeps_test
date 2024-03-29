# Oil Seep Detection using Deep Convolutional Neural Network (DCNN)

## Overview
This project aims to detect oil seeps in synthetic aperture radar (SAR) images using a deep convolutional neural network (DCNN). The provided data consists of SAR images and corresponding masks, where each pixel is classified as either non-seep or one of seven classes of seeps.

## Dataset

The data is a set of synthetic aperture radar (SAR) images chosen at various locations. The images
are 256 x 256 pixels and each pixel is classified as non-seep (0) or 7 classes of seeps (1-7). The SAR
images and their corresponding masks are saved as .tif files with the same names, but in separate
folders, train_images_256/ and train_masks_256/. The objective of the exercise is to segment
regions that contain seeps, and as an optional task to classify the seeps.
For this excercise, I have wroked only on image segmentation but training the unet for multi class should be pretty straighforward, of course I would have to find a way to deal with class imbalance.

### Augmentation
To beef up our dataset without, we can try around with simple tweaks like flipping and rotating our images. These changes can make our limited data go a lot further. However, resizing isn't the best move here because our data is sensitive to where things are placed and how big they are. Plus, DCNN's is pretty good at handling stuff that's moved around, so keeping the original size and shape is key to making sure we don't mess with the important details.

## Model Architecture
I have explored literature on image segmentation on SAR images and stumbled upon a couple of architecutres which might give good results.

1. [U-Net](https://arxiv.org/abs/1505.04597) is great at pinpointing details and understanding the bigger picture, making it suitable for finding oil spills in SAR images. It's good at seeing the difference between oil spills and other stuff by looking at the area around them.

2. [FCNN](https://arxiv.org/abs/1903.11816) excels at creating detailed segmentation maps of any size, thanks to its ability to work with images of varying dimensions.

3. [DeepLab](https://arxiv.org/abs/1606.00915) balances being quick and accurate. They're designed to work fast without compromising on catching all the details, but since the dataset is small this might be the best architecture choice for this use case.

I have chose U-Net as the architecture because it combines detail with a broad view, essential for distinguishing spills from other liquids. Its focus on both local and larger-scale features and as the data set is relatively small it makes sense to use a simple architecture.

## Optimizer
I've chosen the Adam optimizer for its effectiveness in image segmentation, particularly for its ability to adapt learning rates, enhancing the detection of subtle features in complex and noisy datasets like SAR oil seeps, with minimal need for hyperparameter tuning. While Adam provides a user-friendly starting point due to its fewer hyperparameters, exploring Stochastic Gradient Descent (SGD) might be beneficial. With careful hyperparameter tuning, SGD has the potential to offer improved performance by leveraging its simplicity and possibly providing more control over the learning process.



## Evaluation metrics
Found an [aritcle](https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99639aa2) on different methods to evaluate segmentation models.

1. Pixel Accuracy is simple but can mislead in imbalanced datasets, favoring dominant classes without considering spatial distribution. 

$\quad \quad \quad \quad \text{Pixel Accuracy} = \frac{\text{Number of Correctly Classified Pixels}}{\text{Total Number of Pixels}}$

2. Intersection-Over-Union measures the overlap between the predicted segmentation and the ground truth over their union, offering a robust metric that mitigates class imbalance and emphasizes the model's precision in identifying relevant objects.

$\quad \quad \quad \quad \text{IoU} = \frac{\text{Area of Overlap between Prediction and Ground Truth}}{\text{Area of Union between Prediction and Ground Truth}}$

3. Dice Coefficient measures the overlap between the predicted segmentation and the ground truth over their union, offering a robust metric that mitigates class imbalance and emphasizes the model's precision in identifying relevant objects.

$\quad \quad \quad \quad \text{Dice Coefficient} = \frac{2 \times \text{Area of Overlap between Prediction and Ground Truth}}{\text{Total Area of Prediction} + \text{Total Area of Ground Truth}}$


IoU balances penalization of false positives and negatives, while the Dice Coefficient may be more lenient for models predicting larger areas than actual seeps. For this dataset, I have deciced to use both IoU and Dice Coefficient. They account for class imbalance and the spatial nature of segmentation, helping us understand the model performance better. 

Using both seemed like a good option.


## Loss
Found this [paper](https://arxiv.org/pdf/2006.14822.pdf), which does a survey on loss functions for semantic segmentation.

![loss](https://github.com/Harshad165/seeps_test/blob/main/images/Loss_Fn.png)

From these I think DiceBCELoss and FocalTversky Loss are the most suitable for SAR oil seep detection as they offer a effective approach to learning from imbalanced data and focus on critical hard-to-detect features.

1. DiceBCELoss Combines the strengths of Dice loss (focusing on class overlap, useful for imbalanced data) and Binary Cross Entropy (BCE) loss (ensuring good pixel-wise classification), making it effective for balanced learning in scenarios like oil seep detection.

$\quad \quad \quad \quad \text{DiceBCELoss} = \text{BCELoss} - \log\left(\frac{2 \times \text{TP}}{2 \times \text{TP} + \text{FP} + \text{FN}}\right)$


2. FocalTversky Loss is tailored for hard examples and imbalanced classes by adjusting the focus on small segmentation masks, enhancing detection of minor features such as oil seeps. The flexibility in handling false positives and false negatives makes it particularly useful.

$\quad \quad \quad \quad \text{FocalTversky Loss} = (1 - \text{Tversky Index})^\gamma$, $\text{Tversky Index} = \frac{\text{TP}}{\text{TP} + \alpha \times \text{FP} + \beta \times \text{FN}}$

## Regularization

Used Early stopping with a patience of 10 for regularization

## Results

Please take a look at the jupyter notebook for results

## Setup
```
Clone this repo using 
git clone https://github.com/Harshad165/seeps_test.git
```

## Training
To reproduce the training results, run the following command:
```bash
Running the training loop reproduces the saved model. Please check the jupyter notebook for more information
```

## Inference 
```
Please scroll down to the end of the notebook to run inference
```
