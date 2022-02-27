# Main file for the whole workflow of this project. This file will manipulate the data,
# train/cluster features, and analyze potential results.

from data_preparation.DataHelpers import *
from data_preparation.FeatureExtractor import *
from data_preparation.Preprocessor import *
from data_preparation.DimensionalityReducer import *
from learning.Classifier import *
from analysis.Evaluation import *

def prepare_data(data,
                 feature_extractor: FeatureExtractor = None, 
                 preprocessor: Preprocessor = None,
                 dimensionality_reducer: DimensionalityReducer = None):
    if feature_extractor:
        data = feature_extractor.get_features(data)
    if preprocessor:
        data = preprocessor.get_preprocessed_data(data)
    if dimensionality_reducer:
        data = dimensionality_reducer.get_reduced_data(data)

    return data

if __name__ == '__main__':
    train_path = get_training_image_path()
    test_path  = get_test_image_path()

    data = None

    feature_extractor = None
    preprocessor = None
    dimensionality_reducer = None
    # dimensionality_reducer = Sklearn_PCA(n_components=100)

    data_prepared = prepare_data(data, feature_extractor, preprocessor, dimensionality_reducer)
