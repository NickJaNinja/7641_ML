# Classes for classifying features. Most or all supervised learning methods should be here.

from abc import ABC, abstractmethod

class Classifier(ABC):
    def __init__(self, data):
        self.data = data

    @abstractmethod
    def classify(self):
        pass

# TODO implement classifiers like k-Nearest-Neighbors (KNN),
# Support Vector Machines (SVM), Conv Neural Nets (CNN), etc.
