# Main file for the whole workflow of this project. This file will manipulate the data,
# train/cluster features, and analyze potential results.

from data_preparation.FeatureExtractor import *
from data_preparation.Preprocessor import *
from data_preparation.DimensionalityReducer import *
from Classifier import *
from Evaluation import *

# import numpy as np
# import cv2

DATA_DIR = 'data/'

def prepare_data(data,
                 featureExtractor: FeatureExtractor = None, 
                 preprocessor: Preprocessor = None,
                 dimensionalityReducer: DimensionalityReducer = None):
    if featureExtractor:
        data = featureExtractor.get_features(data)
    if preprocessor:
        data = preprocessor.get_preprocessed_data(data)
    if dimensionalityReducer:
        data = dimensionalityReducer.get_reduced_data(data)

    return data

if __name__ == '__main__':
    pass