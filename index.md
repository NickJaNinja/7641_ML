
# CS 7641 Course Project

## Proposal Video

<iframe width="560" height="315" src="https://www.youtube.com/embed/VoxGg14EXxU" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Introduction

According to the WHO, there are approximately 1.3 million people losing their life each year due to road traffic crashes<sup>[4]</sup>. New data released by the USDOT shows that around 20,160 people died in motor vehicle crashes in the first half of 2021<sup>[5]</sup>. Distracted driving is one of the leading factors that cause road traffic crashes. Many researchers are dedicated to detecting distracted behaviors by analyzing image features, hoping to provide more preventive measures. The studies would also benefit the auto-driving industry because the current self-driving cars still require the drivers’ full attention to be able to take back control of the wheels when necessary.

## Problem Statement

Given a set of images of drivers from [Kaggle](https://www.kaggle.com/c/state-farm-distracted-driver-detection), can we correctly classify each image as an accurate level of safety? It might be relatively easier to classify two completely different activities, such as texting and drinking water. However, touching hair and making phone calls could look very similar sometimes. Then what could we do to accurately differentiate similar gestures? What are the metrics we will use to evaluate the accuracy and confidence level? Last but not least, can we improve the computational efficiency relative to the current studies?

## Data Collection

Data was collected through Kaggle. We are given driver images, each taken in a car with a driver doing something, for example: texting, eating, etc. Our goal is to predict the likelihood of what the driver is doing in each image. 

The given training set and testing set were split already for the competition's purpose. There are 22,424 images in the training set and 79,726 images in the testing set. The testing set is much larger, and in order to avoid hand labeling, they included resized images that don’t count towards the competition score. 

The Kaggle competition uses multi-class logarithmic loss to evaluate the submission, and we only used the testing set to get the log-loss score from Kaggle as one of our evaluation methods. For personal evaluations separate from the competition, since we need labeled data for testing purposes, we separately split the training set into 80:20 training-testing ratio.

## Feature Extraction

To reduce the computational cost, we tried out two feature extraction techniques: Histogram of Gradients (HOG) and Scale-Invariant Feature Transform (SIFT/SURF).

### HOG - Histogram of Gradients

As the name indicates, HOG computes pixel-wise gradients and orientations, and plots them on a histogram. It simplifies the representation of images by minimizing noise and capturing only the higher-level information. 

In our case, we used the skimage library in Python to perform HOG and to visualize it. We re-sized all the images to 480 x 640, and then we divided each image into 16x16 patches to extract the features, by setting pixels_per_cell=(16,16), and cells_per_block=(1, 1).

After HOG, the number of features of each image was reduced from 307,200 to 9,600. Below is a visual example to show how an image looks like before and after HOG.

![hog-image](https://user-images.githubusercontent.com/33321452/161834320-07b9a53a-00f6-455d-9ff2-4e48755ef425.png)

### SIFT/SURF - Scale-Invariant Feature Transform

SIFT is another feature extraction method. It picks out keypoints in an image by searching for local extrema after computing the difference of gaussian of the input image. Once keypoints are selected, it creates descriptors based on 8-bin orientation histograms of the keypoint neighborhood grid.

![SIFT](https://user-images.githubusercontent.com/33321452/161837472-39a9d082-d5f1-455a-b7e2-55d634e43d0c.png)

Each descriptor has 128 values. However, the number of keypoints each image has is different. We use K-means to cluster the descriptors and convert them into Bag of Words representations. Finally we feed them into SVM for classification.

## Dimensionality Reduction

### PCA - Principal Component Analysis

To further reduce the dimensionality, we performed PCA after HOG. 

The below plot shows the cumulative sum of explained variance for our dataset. The x-axis represents the number of components, and the y-axis represents the cumulative explained variance (0-1). The dimensionality is now reduced to 224 for each original image. 
The tools used were scikit-learn and plotly.

<img width="761" alt="PCA" src="https://user-images.githubusercontent.com/33321452/161834895-0e28fc2c-6ccd-4b54-9492-cf3685877c83.png">

## Methods

We would like to approach this problem in three ways. First, we start with simpler methods including SVM and KNN. We then apply HOG, SURF to extract features from these images and classify or cluster based on them. Finally, we would train on CNN models to automatically extract and learn features from the dataset.
	
### Unsupervised Learning
We will perform Principal Component Analysis on our data to gain insight of the datasets features and potentially reduce the dimensionality of our data. Given that our data is 640x480 pixel images, dimensionality reduction can improve training computation time while retaining most of the information<sup>[3]</sup>. Clustering algorithms like K-means will also give us insight on the dataset.

### Supervised Learning
The nature of our data is unimodal and our intuition tells us that the best results should come from models that are best able to understand human gestures. Among the many human activity recognition techniques, we will focus on the ones better for learning human poses and interactions with objects<sup>[1]</sup>. We will pre-process data and train on popular CNN architectures like VGG, ResNet.

## Evaluation

The Kaggle competition-defined loss is as below, where N is the number of images in the test set and M is the number of image class labels:

<img src="https://render.githubusercontent.com/render/math?math=logloss = -\frac{1}{N} \sum_{i=1}^{N}\sum_{j=1}^{M}y_{ij}\log(p_{ij})">

The first supervised method we have results for is from support-vector machines (SVM). We used Sklearn for this, which handles multiclass classification using one-vs-one as opposed to one-vs-rest. Although SVM is known for robustness, training the classifier on our entire unprocessed dataset was infeasible due to its size. Therefore, we used HOG to compress the data before training this classifier. Our results from this method gave us a Kaggle log loss of 1.68050. This score of our baseline method puts us at 725th place of the competition, but we’re hoping our more sophisticated convolutional neural nets (CNNs) will see a boost in our rating.

## Discussion

After obtaining valuable features by PCA, we will conduct several experiments based on the different methods mentioned above. The classification result will be soft, so we will take the max of our predicted output to get the most likely class, which is used to measure the F1 score and cross-entropy loss. 

Based on our findings, the difference in performance between models with proper parameters optimization should be round 0.01 magnitude. Also, data leakage may lead to a high performance, which means the same person (but slight changes of angle or shifts in height or width) within a class may appear in both training and testing<sup>[6]</sup>. To avoid this, we will split the images based on the person IDs.

## Timeline

![image](https://user-images.githubusercontent.com/33321452/154896439-96ad843c-6ecc-49f2-9ca7-edd7bf91effc.png)

## References
1. Vrigkas, M., Nikou, C., & Kakadiaris, I. A. (2015). A review of human activity recognition methods. Frontiers in Robotics and AI, 2, 28.
2. Mafeni Mase, J., Chapman, P., Figueredo, G. P., & Torres Torres, M. (2020, July). Benchmarking deep learning models for driver distraction detection. In International Conference on Machine Learning, Optimization, and Data Science (pp. 103-117). Springer, Cham.
3. [https://ai.plainenglish.io/distracted-driver-detection-using-machine-and-deep-learning-techniques-1ba7e7ce0225](https://ai.plainenglish.io/distracted-driver-detection-using-machine-and-deep-learning-techniques-1ba7e7ce0225)
4. [https://www.who.int/news-room/fact-sheets/detail/road-traffic-injuries](https://www.who.int/news-room/fact-sheets/detail/road-traffic-injuries)
5. [https://www.nhtsa.gov/press-releases/usdot-releases-new-data-showing-road-fatalities-spiked-first-half-2021](https://www.nhtsa.gov/press-releases/usdot-releases-new-data-showing-road-fatalities-spiked-first-half-2021)
6. [https://towardsdatascience.com/distracted-driver-detection-using-deep-learning-e893715e02a4 ](https://towardsdatascience.com/distracted-driver-detection-using-deep-learning-e893715e02a4 )
