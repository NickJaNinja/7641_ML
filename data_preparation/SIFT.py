from data_preparation.FeatureExtractor import *

import numpy as np
import cv2

class SIFT(FeatureExtractor):
    def __init__(self):
        pass

    def get_features(self, image):
        extractor = cv2.xfeatures2d.SIFT_create()
        keypoints = extractor.detect(image, None)
        self.keypoints = keypoints
        return keypoints

    def get_features_batch(self, data):
        features = [self.get_features(image) for image in data]
        return features
