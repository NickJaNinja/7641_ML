from data_preparation.DimensionalityReducer import *

import numpy as np
from sklearn import decomposition

class Sklearn_PCA(DimensionalityReducer):
    def __init__(self, n_components):
        """
        Args: 
            data: NxD array, where each image is flattened into a 1-Dimensional vector
            n_components (int): Number of components/features for PCA to reduce to
        """
        self.n_components = n_components

    def get_reduced_data(self, data):
        """Performs PCA on the data.
        Sklean's PCA assumes the data is size NxD, where N is the number of images,
        and D is the dimension of the *flattened* image.
        """
        num_images = data.shape[0]
        data = np.reshape(data, (num_images, -1))
        pca = decomposition.PCA(self.n_components)
        pca.fit(data)
        return pca.transform(data)
