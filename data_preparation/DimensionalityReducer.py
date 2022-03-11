# Reduces number of features by dimensionality reduction.

from abc import ABC, abstractmethod

# EXAMPLE USE
# pca = PCA(data, n_components)
# new_data = pca.get_reduced_data()

class DimensionalityReducer(ABC):
    @abstractmethod
    def get_reduced_data(self):
        pass
