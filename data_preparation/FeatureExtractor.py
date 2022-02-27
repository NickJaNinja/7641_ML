# Any feature extraction method to help a downstram clustering or classification.

from abc import ABC, abstractmethod

class FeatureExtractor(ABC):
    @abstractmethod
    def get_features(data):
        pass


# TODO: add new FeatureExtractor classes like HOG, SURF, etc.
