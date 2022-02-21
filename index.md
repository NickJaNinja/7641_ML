
# CS 7641 Course Project

## Proposal 

### Introduction

According to the WHO, there are approximately 1.3 million people losing their life each year due to road traffic crashes<sup>[4]</sup>. New data released by the USDOT shows that around 20,160 people died in motor vehicle crashes in the first half of 2021<sup>[5]</sup>. Distracted driving is one of the leading factors that cause road traffic crashes. Many researchers are dedicated to detecting distracted behaviors by analyzing image features, hoping to provide more preventive measures. The studies would also benefit the auto-driving industry because the current self-driving cars still require the drivers’ full attention to be able to take back control of the wheels when necessary.

### Problem Statement

Given a set of images of drivers from [Kaggle](https://www.kaggle.com/c/state-farm-distracted-driver-detection), can we correctly classify each image as an accurate level of safety? It might be relatively easier to classify two completely different activities, such as texting and drinking water. However, touching hair and making phone calls could look very similar sometimes. Then what could we do to accurately differentiate similar gestures? What are the metrics we will use to evaluate the accuracy and confidence level? Last but not least, can we improve the computational efficiency relative to the current studies?

### Methods

We would like to approach this problem in three ways. First, we start with simpler methods including SVM and KNN. We then apply HOG, SURF to extract features from these images and classify or cluster based on them. Finally, we would train on CNN models to automatically extract and learn features from the dataset.
	
#### Unsupervised Learning
We will perform Principal Component Analysis on our data to gain insight of the datasets features and potentially reduce the dimensionality of our data. Given that our data is 640x480 pixel images, dimensionality reduction can improve training computation time while retaining most of the information<sup>[3]</sup>. Clustering algorithms like K-means will also give us insight on the dataset.

#### Supervised Learning
The nature of our data is unimodal and our intuition tells us that the best results should come from models that are best able to understand human gestures. Among the many human activity recognition techniques, we will focus on the ones better for learning human poses and interactions with objects<sup>[1]</sup>. We will pre-process data and train on popular CNN architectures like VGG, ResNet.

### Discussion

After obtaining valuable features by PCA, we will conduct several experiments based on the different methods mentioned above. The classification result will be soft, so we will take the max of our predicted output to get the most likely class, which is used to measure the F1 score and cross-entropy loss. 

Based on our findings, the difference in performance between models with proper parameters optimization should be round 0.01 magnitude. Also, data leakage may lead to a high performance, which means the same person (but slight changes of angle or shifts in height or width) within a class may appear in both training and testing<sup>[6]</sup>. To avoid this, we will split the images based on the person IDs.

### Timeline

![image](https://user-images.githubusercontent.com/33321452/154896439-96ad843c-6ecc-49f2-9ca7-edd7bf91effc.png)

### References
1. Vrigkas, M., Nikou, C., & Kakadiaris, I. A. (2015). A review of human activity recognition methods. Frontiers in Robotics and AI, 2, 28.
2. Mafeni Mase, J., Chapman, P., Figueredo, G. P., & Torres Torres, M. (2020, July). Benchmarking deep learning models for driver distraction detection. In International Conference on Machine Learning, Optimization, and Data Science (pp. 103-117). Springer, Cham.
3. [https://ai.plainenglish.io/distracted-driver-detection-using-machine-and-deep-learning-techniques-1ba7e7ce0225](https://ai.plainenglish.io/distracted-driver-detection-using-machine-and-deep-learning-techniques-1ba7e7ce0225)
4. [https://www.who.int/news-room/fact-sheets/detail/road-traffic-injuries](https://www.who.int/news-room/fact-sheets/detail/road-traffic-injuries)
5. [https://www.nhtsa.gov/press-releases/usdot-releases-new-data-showing-road-fatalities-spiked-first-half-2021](https://www.nhtsa.gov/press-releases/usdot-releases-new-data-showing-road-fatalities-spiked-first-half-2021)
6. [https://towardsdatascience.com/distracted-driver-detection-using-deep-learning-e893715e02a4 ](https://towardsdatascience.com/distracted-driver-detection-using-deep-learning-e893715e02a4 )
