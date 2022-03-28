from learning.Classifier import *

from sklearn import svm

class Sklearn_SVM(Classifier):
    def __init__(self, kernel='linear'):
        self.classifier = svm.SVC(kernel=kernel, probability=True)

    def train(self, x_train, y_train):
        self.classifier.fit(x_train, y_train)

    def predict(self, x_predict):
        return self.classifier.predict_proba(x_predict)
