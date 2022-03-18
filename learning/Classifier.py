# Classes for classifying features. Most or all supervised learning methods should be here.

from abc import ABC, abstractmethod

class Classifier(ABC):
    @abstractmethod
    def train(self, x_train, y_train):
        pass

    @abstractmethod
    def predict(self, x_predict):
        pass

# TODO implement classifiers like k-Nearest-Neighbors (KNN),
# Support Vector Machines (SVM), Conv Neural Nets (CNN), etc.
