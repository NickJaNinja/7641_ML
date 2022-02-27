# Reduces number of features by dimensionality reduction.

from abc import ABC, abstractmethod

from sklearn import decomposition

# EXAMPLE USE
# pca = PCA(data, n_components)
# new_data = pca.get_reduced_data()

class DimensionalityReducer(ABC):
    def __init__(self, data):
        self.data = data

    @abstractmethod
    def get_reduced_data(self):
        pass


class PCA(DimensionalityReducer):
    def __init__(self, data, n_components):
        self.data = data
        self.n_components = n_components

    def get_reduced_data(self):
        pca = decomposition.PCA(self.n_components)
        return pca.fit(self.data)
