from data_preparation.FeatureExtractor import *

import numpy as np
import cv2

class SURF(FeatureExtractor):
    def __init__(self, hessian_threshold=800):
        self.hessian_threshold = hessian_threshold

    def get_features(self, image):
        extractor = cv2.xfeatures2d.SURF_create(self.hessian_threshold)
        keypoints, descriptor = extractor.detectAndCompute(image, None)
        self.keypoints = keypoints
        return descriptor.flatten()

    def get_features_batch(self, data):
        features = [self.get_features(image) for image in data]
        return features
