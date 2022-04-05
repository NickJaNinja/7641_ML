from data_preparation.FeatureExtractor import *
from data_preparation.BOW import BOW

import numpy as np
import cv2

class ORB(FeatureExtractor):
    def __init__(self, N=80):
        self.kmeans_clusters = N
        self.extractor = cv2.ORB_create()

    def get_features_single(self, image, visualize=False):
        keypoints, descriptors = self.extractor.detectAndCompute(image, None)
        if visualize:
            image_kp = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            return image_kp, descriptors
        else:
            return descriptors

    def get_features(self, data):
        descriptors = [self.get_features_single(image) for image in data]
        bag_of_descriptors = np.concatenate(descriptors)
        bow = BOW(bag_of_descriptors, self.kmeans_clusters)
        features = bow.predict(descriptors)
        return features
