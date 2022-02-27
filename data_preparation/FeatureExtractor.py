# Any feature extraction method to help a downstram clustering or classification.

from abc import ABC, abstractmethod

class FeatureExtractor(ABC):
    def __init__(self, data):
        self.data = data

    @abstractmethod
    def get_features(data):
        pass


# TODO: add new FeatureExtractor classes like HOG, SURF, etc.
