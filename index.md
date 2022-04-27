
# CS 7641 Course Project

## Proposal Video

[Link to video](https://www.youtube.com/watch?v=VoxGg14EXxU&ab_channel=SarahGe)

## Final Video

<iframe width="560" height="315" src="https://www.youtube-nocookie.com/embed/aCEr_zM4t5s" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Introduction

According to the WHO, there are approximately 1.3 million people losing their life each year due to road traffic crashes<sup>[4]</sup>. New data released by the USDOT shows that around 20,160 people died in motor vehicle crashes in the first half of 2021<sup>[5]</sup>. Distracted driving is one of the leading factors that cause road traffic crashes. Many researchers are dedicated to detecting distracted behaviors by analyzing image features, hoping to provide more preventive measures. The studies would also benefit the auto-driving industry because the current self-driving cars still require the drivers’ full attention to be able to take back control of the wheels when necessary.

## Problem Statement

Given a set of images of drivers from [Kaggle](https://www.kaggle.com/c/state-farm-distracted-driver-detection), can we correctly classify each image as an accurate level of safety? It might be relatively easier to classify two completely different activities, such as texting and drinking water. However, touching hair and making phone calls could look very similar sometimes. Then what could we do to accurately differentiate similar gestures? What are the metrics we will use to evaluate the accuracy and confidence level? Last but not least, can we improve the computational efficiency relative to the current studies?

## Data Collection 

Data was collected through Kaggle. We are given driver images, each taken in a car with a driver doing something, for example: texting, eating, etc. Our goal is to predict the likelihood of what the driver is doing in each image.

![10_classes](https://user-images.githubusercontent.com/33321452/165432051-0996e162-a85e-4987-84aa-5345420ec01e.jpg)

The 10 classes are:
- c0: safe driving
- c1: texting - right
- c2: talking on the phone - right
- c3: texting - left
- c4: talking on the phone - left
- c5: operating the radio
- c6: drinking
- c7: reaching behind
- c8: hair and makeup
- c9: talking to passenger

The given training set and testing set were split already for the competition's purpose. There are 22,424 images in the training set and 79,726 images in the testing set. The testing set is much larger, and in order to avoid hand labeling, they included resized images that don’t count towards the competition score.

The Kaggle competition uses multi-class logarithmic loss to evaluate the submission, and we only used the testing set to get the log-loss score from Kaggle as one of our evaluation methods. For personal evaluations separate from the competition, since we need labeled data for testing purposes, we separately split the training set into 80:20 training-testing ratio.

## Methods

We would like to approach this problem in three ways. First, we start with simpler methods including SVM and KNN. We then apply HOG, SURF to extract features from these images and classify or cluster based on them. Finally, we would train on CNN models to automatically extract and learn features from the dataset.
	
### Unsupervised Learning
We will perform Principal Component Analysis on our data to gain insight of the datasets features and potentially reduce the dimensionality of our data. Given that our data is 640x480 pixel images, dimensionality reduction can improve training computation time while retaining most of the information<sup>[3]</sup>. Clustering algorithms like K-means will also give us insight on the dataset.

### Supervised Learning
The nature of our data is unimodal and our intuition tells us that the best results should come from models that are best able to understand human gestures. Among the many human activity recognition techniques, we will focus on the ones better for learning human poses and interactions with objects<sup>[1]</sup>. We will pre-process data and train on popular CNN architectures like VGG, ResNet, EfficientNet, and Xception.

## Feature Extraction

To reduce the computational cost, we tried out two feature extraction techniques: Histogram of Gradients (HOG) and Scale-Invariant Feature Transform (SIFT/SURF).

### 1. HOG - Histogram of Gradients

As the name indicates, HOG computes pixel-wise gradients and orientations, and plots them on a histogram. It simplifies the representation of images by minimizing noise and capturing only the higher-level information. 

In our case, we used the skimage library in Python to perform HOG and to visualize it. We re-sized all the images to 480 x 640, and then we divided each image into 16x16 patches to extract the features, by setting pixels_per_cell=(16,16), and cells_per_block=(1, 1).

After HOG, the number of features of each image was reduced from 307,200 to 9,600. Below is a visual example to show how an image looks like before and after HOG.

![hog-image](https://user-images.githubusercontent.com/33321452/161834320-07b9a53a-00f6-455d-9ff2-4e48755ef425.png)

### 2. SIFT/SURF - Scale-Invariant Feature Transform

SIFT is another feature extraction method. It picks out keypoints in an image by searching for local extrema after computing the difference of gaussian of the input image. Once keypoints are selected, it creates descriptors based on 8-bin orientation histograms of the keypoint neighborhood grid.

![SIFT](https://user-images.githubusercontent.com/33321452/161837472-39a9d082-d5f1-455a-b7e2-55d634e43d0c.png)

Each descriptor has 128 values. However, the number of keypoints each image has is different. We use K-means to cluster the descriptors and convert them into Bag of Words representations. Finally we feed them into SVM for classification.

## Dimensionality Reduction

### 1. PCA - Principal Component Analysis

To further reduce the dimensionality, we performed PCA after HOG. 

The below plot shows the cumulative sum of explained variance for our dataset. The x-axis represents the number of components, and the y-axis represents the cumulative explained variance (0-1). The dimensionality is now reduced to 224 for each original image. 
The tools used were scikit-learn and plotly.

<img width="761" alt="PCA" src="https://user-images.githubusercontent.com/33321452/161834895-0e28fc2c-6ccd-4b54-9492-cf3685877c83.png">

## Unsupervised Modeling

### K-Means

We use K-means on 2500 flattened images (250 from each category). We set k=10 to represent the 10 classes of driving safety in the dataset.  Because we took 250 from each group sequentially, we can easily visualize the result with the following histogram showing the number of images per cluster (as given by the k-means algorithm).

![Kmeans_1](https://user-images.githubusercontent.com/33321452/161884313-dc115c73-0f1f-4fee-adcc-a3696341a1d9.png)

Ideally, we would see 250 images in each cluster and we could verify the truth value of such a cluster by its image order. Unfortunately, the Kmeans clustering does a poor job at clustering each of the 10 categories. We see over-representation of cluster 3 with nearly twice the number of images as should be associated with this cluster. Clusters 1, 6, and 9 are under-represented each with less than 200 images associated with their respective clusters. This is not as informative as we would like, but because we are using labeled data, we can look further into why we are receiving these results.

The following plot shows the mode cluster for each of the ten categories:

![mode_cluster](https://user-images.githubusercontent.com/33321452/165432708-8b2d5ae4-3a14-4ced-a87f-e344a90b12c6.jpeg)

To explain further, 9 of the 10 categories were most often assigned cluster 3. Cluster 7 was the mode of c0 (the case where the drivers were not distracted). Although it is interesting that our non-distracted class seems to be separated from our distracted class, it is important to note that on average, the number of images constituting the mode was only around 50 images for all modes (or around 20%). This means that with each group of 250 images, the images are not clearly associated with one cluster over another.  Overall, our finding could be improved by adding more images, or separating the k-means by color channel, but we would like to move on to more promising approaches.

## Supervised Modeling

### 1. SVM - Support Vector Machines

The first supervised method we have results for is from support-vector machines (SVM). We used Sklearn for this, which handles multiclass classification using one-vs-one as opposed to one-vs-rest. Although SVM is known for robustness, training the classifier on our entire unprocessed dataset was infeasible due to its size. Therefore, we used HOG to compress the data before training this classifier. Our results from this method gave us a Kaggle log loss of 1.68050. This score of our baseline method puts us at 725th place of the competition, but we’re hoping our more sophisticated convolutional neural nets (CNNs) will see a boost in our rating.

The Kaggle competition-defined loss is as below, where N is the number of images in the test set and M is the number of image class labels:

<img src="https://render.githubusercontent.com/render/math?math=logloss = -\frac{1}{N} \sum_{i=1}^{N}\sum_{j=1}^{M}y_{ij}\log(p_{ij})">

Besides the internal Kaggle evaluation, we used macro-average-F1-score as a metric to evaluate our SVM model. Due to the huge size of the dataset, we evaluated our model with gradually increasing numbers of training and testing data points. Results from training on 10% of the data are as follows:

![f1_2_1](https://user-images.githubusercontent.com/33321452/161892542-ef2304e5-7736-4e35-9178-6f3d8b4a0c76.png)

![f1_2_2](https://user-images.githubusercontent.com/33321452/161892554-79d7f323-6ef9-4757-8bdf-4089dd8e46ab.png)

![f1_2_3](https://user-images.githubusercontent.com/33321452/161892571-e0e835a5-cb3b-4312-84dd-b62ed51ada60.png)

The score (96.93%) is high, which makes us aware that the data leakage problem is indeed occurring. The testing set likely contains images that are almost identical to the training set because of the quality of the given dataset. Thus, we will explore more reasonable evaluation methods other than Kaggle’s in the future.

### 2. CNN Models

Lastly, we show results for some deep learning methods.

#### Baseline Model

We start by creating a baseline model and training from scratch. It consists of three convolution blocks of filter size (5,5), each with increasing depth. We apply maxpool and batch normalization after each convolution.

| ![baseline_cnn](https://user-images.githubusercontent.com/33321452/165433230-1e72429e-9a5f-4546-a227-a0507aa32fb5.png) |
|:--:|
| *Baseline Architecture* |

![baseline_results](https://user-images.githubusercontent.com/33321452/165433489-523020e4-2652-4ab5-baad-dafcb76bd953.png)

#### Transfer Learning

Since CNN (Convolutional Neural Networks) are known as the current best algorithms for image processing, we adopted a few most popular pre-trained CNN architectures, fine-tuned hyper parameters, and compared results. The architectures we used include: [EfficientNet(B0,B5)](https://ai.googleblog.com/2019/05/efficientnet-improving-accuracy-and.html), [ResNet50](https://blog.devgenius.io/resnet50-6b42934db431), [Xception](https://towardsdatascience.com/review-xception-with-depthwise-separable-convolution-better-than-inception-v3-image-dc967dd42568), and [VGG16](https://www.kaggle.com/code/blurredmachine/vggnet-16-architecture-a-complete-guide/notebook).

For all models, we started with using the ImageNet weights and trained only the top layers. All models achieved decent performance. The validation accuracies were between 0.7612 ~ to 0.9906.

Then we tried to train additional layers for EfficientNetB0 and Xception. We trained full layers for EfficientNetB0, and we observed an increase in the validation accuracy by 7%. We trained top layers and the last convolutional block for Xception, and we observed an increase in validation accuracy by 14%.

For VGG we trained the top layers and the last convolutional block, without and with data augmentation. We found that with data augmentation, the model generalized better and achieved better validation accuracy.

Using the Log-loss score from Kaggle as the evaluation metrics, EfficientNetB0 trained from scratch had a score of 1.02171, which outperformed all other models.

#### CNN Performance

![all_results](https://user-images.githubusercontent.com/33321452/165433897-0534f82a-7833-4e3b-869e-4874945e01a6.jpg)

#### Accuracy and Loss of the Models

![efficientnetb0_pretrained_results](https://user-images.githubusercontent.com/33321452/165433932-479d3c71-9564-479f-ac1f-4b6b6848d2a3.png)

![efficientnetb0_fromscratch_results](https://user-images.githubusercontent.com/33321452/165433937-75af4cff-ed1f-4e11-8d4e-ebab24f8cb97.png)

![efficientnetb5_pretrained_results](https://user-images.githubusercontent.com/33321452/165433943-a13a67a2-0e3c-494f-ae7e-ceb28f8e6787.png)

![resnet50_fromscratch_results](https://user-images.githubusercontent.com/33321452/165433955-84c4cd13-be0b-4bbb-b7c6-80440e98d24e.png)

![xception_worse](https://user-images.githubusercontent.com/33321452/165433983-60eb78f2-3eb1-48ae-a10c-b5e9fdc04f83.png)

![xception_better](https://user-images.githubusercontent.com/33321452/165433988-76b4519c-3eea-4a48-a0a6-d5ef232ab012.png)

![vgg_nodata](https://user-images.githubusercontent.com/33321452/165434006-c4031a72-c79f-4434-b511-36cd6a87e4cc.png)

![vgg_data](https://user-images.githubusercontent.com/33321452/165434012-190f2810-a6ec-4bb8-9260-d3d3a56e98fb.png)

## Conclusion

We experimented with various models to cluster or classify the images. For non-deep-learning approaches, we evaluated the model performance by using the Log-loss score provided by Kaggle, and the macro-average F1 score. For the CNN models, the evaluation metrics include: training and testing accuracy, training and testing loss, and the Log-loss score.

Overall the pre-trained CNN models all performed decently, but there are many other things to think about and try out. For example: which hyperparameters are more sensitive? How to keep fine-tuning the hyperparameters to get better generalization? Which layers should we retrain? Which optimizer to use? How can we deploy to the end application?

## Timeline

![image](https://user-images.githubusercontent.com/33321452/154896439-96ad843c-6ecc-49f2-9ca7-edd7bf91effc.png)

## References
1. Vrigkas, M., Nikou, C., & Kakadiaris, I. A. (2015). A review of human activity recognition methods. Frontiers in Robotics and AI, 2, 28.
2. Mafeni Mase, J., Chapman, P., Figueredo, G. P., & Torres Torres, M. (2020, July). Benchmarking deep learning models for driver distraction detection. In International Conference on Machine Learning, Optimization, and Data Science (pp. 103-117). Springer, Cham.
3. [https://ai.plainenglish.io/distracted-driver-detection-using-machine-and-deep-learning-techniques-1ba7e7ce0225](https://ai.plainenglish.io/distracted-driver-detection-using-machine-and-deep-learning-techniques-1ba7e7ce0225)
4. [https://www.who.int/news-room/fact-sheets/detail/road-traffic-injuries](https://www.who.int/news-room/fact-sheets/detail/road-traffic-injuries)
5. [https://www.nhtsa.gov/press-releases/usdot-releases-new-data-showing-road-fatalities-spiked-first-half-2021](https://www.nhtsa.gov/press-releases/usdot-releases-new-data-showing-road-fatalities-spiked-first-half-2021)
6. [https://towardsdatascience.com/distracted-driver-detection-using-deep-learning-e893715e02a4 ](https://towardsdatascience.com/distracted-driver-detection-using-deep-learning-e893715e02a4 )
