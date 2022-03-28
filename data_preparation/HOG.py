from data_preparation.FeatureExtractor import *

import numpy as np
from skimage.feature import hog

class HOG(FeatureExtractor):
    def __init__(self, orientations=8, pixels_per_cell=(16,16)):
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell

    def get_features(self, data, visualize=False):
        '''
        Args:
            data: N x H x W x D, where N is the number of images
        
        Returns: a tuple of two numpy arrays: (extracted_features, images_after_hog)
            
        '''
        features = []
        hog_images = []
        
        for image in data:
            # Using channel_axis param requires scikit-image version 0.19.2
            flattened_hog = hog(image, self.orientations, self.pixels_per_cell,
                                cells_per_block=(1, 1), visualize=visualize, channel_axis=-1)

            if visualize:
                features.append(flattened_hog[0])
                hog_images.append(flattened_hog[1])
            else:
                features.append(flattened_hog)
    
        if visualize:
            return (np.array(features),np.array(hog_images))
        else:
            return np.array(features)
