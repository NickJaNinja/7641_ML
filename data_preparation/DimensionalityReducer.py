# Reduces number of features by dimensionality reduction.

from abc import ABC, abstractmethod

from sklearn import decomposition

# EXAMPLE USE
# pca = PCA(data, n_components)
# new_data = pca.get_reduced_data()

class DimensionalityReducer(ABC):
    @abstractmethod
    def get_reduced_data(self):
        pass


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
        pca = decomposition.PCA(self.n_components)
        pca.fit(data)
        return pca.transform(data)
