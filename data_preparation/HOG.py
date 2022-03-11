from data_preparation.FeatureExtractor import *

import numpy as np
from skimage.feature import hog

class HOG(FeatureExtractor):
    def __init__(self, orientations=8, pixels_per_cell=(16,16)):
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell

    def get_features(self, data):
        features = []
        for image in data:
            flattened_hog = hog(image, self.orientations,
                    self.pixels_per_cell, channel_axis=-1)
            features.append(flattened_hog)
        return np.array(features)
