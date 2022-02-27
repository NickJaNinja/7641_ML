# Performing preprocessing of data only, does not change dimensionality of features.

from abc import ABC, abstractmethod

class Preprocessor(ABC):
    def __init__(self, data):
        self.data = data

    @abstractmethod
    def get_preprocessed_data(self):
        pass

# TODO: add new Preprocessors classes like mini-max normalization as in
# https://ai.plainenglish.io/distracted-driver-detection-using-machine-and-deep-learning-techniques-1ba7e7ce0225 
