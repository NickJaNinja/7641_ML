# Main file for the whole workflow of this project. This file will manipulate the data,
# train/cluster features, and analyze potential results.

from data_preparation import DataHelpers, FeatureExtractor, Preprocessor, DimensionalityReducer
from data_preparation.HOG import *
from data_preparation.Sklearn_PCA import *

from learning.Sklearn_SVM import *

def prepare_batch(data,
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

def prepare_batches(data_paths, 
                 labels = None,
                 feature_extractor: FeatureExtractor = None, 
                 preprocessor: Preprocessor = None,
                 dimensionality_reducer: DimensionalityReducer = None):

    num_batches = len(data_paths)
    prepared_batches = None

    for batch_i in range(num_batches):
        batch_images = DataHelpers.open_batch(data_paths[batch_i])
        prepared_batch = prepare_batch(batch_images, feature_extractor, preprocessor, dimensionality_reducer)
        if prepared_batches is None:
            prepared_batches = prepared_batch
        else:
            prepared_batches = np.vstack((prepared_batches, prepared_batch))

        if num_batches >= 10 and batch_i % (num_batches//10) == 0 and batch_i > 0:
            print(f'Batch {batch_i}/{num_batches} done preparing...')

    D = prepared_batches.shape[-1]
    data = np.concatenate(prepared_batches, axis=0).reshape((-1, D))

    if labels is not None:
        labels = np.concatenate(labels, axis=0).reshape((-1,))
        return (data, labels)
    else:
        return data

if __name__ == '__main__':
    feature_extractor = HOG(orientations=8, pixels_per_cell=(32,32))
    preprocessor = None
    dimensionality_reducer = None
    # dimensionality_reducer = Sklearn_PCA(n_components=50)

    TRAINING_NEW_CLASSIFIER = False
    if TRAINING_NEW_CLASSIFIER:
        classifier = Sklearn_SVM()

        NUM_BATCHES_TRAIN = 350
        batches_as_paths, batches_labels = DataHelpers.generate_training_batches(NUM_BATCHES_TRAIN)
        x_train, y_train = prepare_batches(batches_as_paths, batches_labels, feature_extractor, preprocessor, dimensionality_reducer)

        classifier.train(x_train, y_train)
        DataHelpers.save_model(classifier, 'hog_svm')
    else:
        classifier = DataHelpers.load_model('hog_svm')

    NUM_BATCHES_TEST = 350
    x_test_names, x_test_paths = DataHelpers.generate_testing_batches(NUM_BATCHES_TEST)
    prepared_test = prepare_batches(x_test_paths, None, feature_extractor, preprocessor, dimensionality_reducer)
    # print(prepared_test.shape)

    y_pred = classifier.predict(prepared_test)

    x_test_names = np.concatenate(x_test_names, axis=0).reshape((-1,))
    DataHelpers.output_to_csv('Sklearn_SVM', x_test_names, y_pred)

    print('DONE!')
