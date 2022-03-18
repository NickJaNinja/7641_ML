# Main file for the whole workflow of this project. This file will manipulate the data,
# train/cluster features, and analyze potential results.

from data_preparation import DataHelpers, FeatureExtractor, Preprocessor, DimensionalityReducer
from data_preparation.HOG import *
from data_preparation.Sklearn_PCA import *

from learning.Sklearn_SVM import *

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
    training_batches_as_paths, labels = DataHelpers.generate_training_batches(num_batches=350)
    train_batch, train_labels = DataHelpers.open_training_batch(training_batches_as_paths[0]), labels[0]
    test_batch, test_labels = DataHelpers.open_training_batch(training_batches_as_paths[1]), labels[1]

    feature_extractor = HOG(orientations=8, pixels_per_cell=(16,16))
    preprocessor = None
    dimensionality_reducer = None
    # dimensionality_reducer = Sklearn_PCA(n_components=50)

    # classifier = Sklearn_SVM()

    train_prepared = prepare_data(train_batch, feature_extractor, preprocessor, dimensionality_reducer)
