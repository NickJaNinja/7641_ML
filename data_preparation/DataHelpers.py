# Functions to assist in fetching data.

import os, os.path
import csv
import pickle
from shutil import copytree, ignore_patterns
import random
from PIL import Image
import numpy as np

DATA_DIR = 'data/'
COMPRESSED_DATA_DIR = 'compressed_' + DATA_DIR
RESIZE_IMAGE_RATIO = 2

if os.path.exists(COMPRESSED_DATA_DIR):
    TRAIN_DIR = COMPRESSED_DATA_DIR + 'train/'
    TEST_DIR = COMPRESSED_DATA_DIR + 'test/'
else:
    TRAIN_DIR = DATA_DIR + 'train/'
    TEST_DIR = DATA_DIR + 'test/'

MODEL_DIR = 'models/'
OUTPUT_DIR = 'output/'
NUM_CLASSES = 10


def _get_training_class_path(driving_class: int = 0):
    """Gets the path to a specific class of the training images
    
    Args:
        driving_class (int): Class from 0 to 9 corresponding to the class of driving

    Returns:
        str: Path to the directory of images
    """
    if driving_class < 0 or driving_class > 9:
        return None
    class_folder_path = f'{TRAIN_DIR}c{driving_class}/'
    return class_folder_path

def _get_training_class_paths():
    """
    Returns:
        list[str]: List of paths to the training class directories.
    """
    return [_get_training_class_path(id) for id in range(NUM_CLASSES)]

def get_training_image_paths_by_class():
    return [[f'{c}{i}' for i in os.listdir(c)] for c in _get_training_class_paths()]

def get_training_image_paths_flat():
    return [f'{c}{i}' for c in _get_training_class_paths() for i in os.listdir(c)]

def _get_test_image_names():
    return [img_name for img_name in os.listdir(TEST_DIR)]

def _get_test_image_paths(image_names):
    """Gets the path to all test images

    Returns:
        str: Path to the directory
    """
    return [f'{TEST_DIR}{img_name}' for img_name in image_names]

def get_class_from_path(path):
    return int(path.split('/')[2][1:])

def open_image_as_np(path_to_img: str):
    """
    Returns:
        np.array: Opened image
    """
    img = Image.open(path_to_img)
    img_as_np = np.asarray(img)
    return img_as_np

def get_random_training_image():
    """Gets a random training image.

    Returns:
        np.array: The random image
    """
    all_image_paths = get_training_image_paths_flat()
    rand_path = random.choice(all_image_paths)
    return open_image_as_np(rand_path)

def get_num_training_images():
    """
    Returns:
        int: Number of training images
    """
    return len(get_training_image_paths_flat())

def generate_training_batches(num_batches=200):
    """Divides training data into batches.

    Returns:
        list[list[str]]: Batched image paths
        list[list[int]]: Label for each corresponding image
    """
    num_images = get_num_training_images()
    image_paths = get_training_image_paths_flat()
    random.shuffle(image_paths)
    batch = np.linspace(0, num_images, num_batches+1, dtype=int)

    batch_paths = [image_paths[batch[i]:batch[i+1]] for i in range(num_batches)]
    batch_labels = [np.array([get_class_from_path(p) for p in b]) for b in batch_paths]
    return batch_paths, batch_labels

def _get_name_from_path(path):
    return path.split('/')[2]

def generate_testing_batches(num_batches=20):
    """Divides training data into batches.

    Returns:
        list[list[str]]: Batched image paths
        list[list[int]]: Label for each corresponding image
    """
    test_image_names = _get_test_image_names()
    image_paths = _get_test_image_paths(test_image_names)
    _get_name_from_path(image_paths[0])
    num_images = len(image_paths)
    random.shuffle(image_paths)
    batch = np.linspace(0, num_images, num_batches+1, dtype=int)

    batch_paths = [image_paths[batch[i]:batch[i+1]] for i in range(num_batches)]
    batch_names = [np.array([_get_name_from_path(p) for p in b]) for b in batch_paths]
    return batch_names, batch_paths

def open_batch(image_paths):
    """Given a batch of paths, opens the whole batch of images and returns them.

    Returns:
        np.array[image]: Batch of images, shape (N x w x h x 3)
    """
    images = []
    for image_path in image_paths:
        img = open_image_as_np(image_path)
        images.append(img)
    images = np.array(images)
    return images

def get_test_data():
    """
    """
    test_image_names = _get_test_image_names()
    test_image_paths = _get_test_image_paths(test_image_names)
    return (test_image_names, test_image_paths)

def output_to_csv(filename, image_names, y_predicted):
    """Produces an output file for the given image names and predicted values.
    The format follows the Kaggle competition's requirement.

    Args:
        filename (str): Name of output file
        image_names (list[str]): (N,1) list of names of each image corresponding to each row in y_predicted
        y_predicted (list[list[int]]): (N,NUM_CLASSES) predicted probability of each class
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    # Change filename if filename already exists to avoid overwriting output
    output_path = f'{OUTPUT_DIR}{filename}.csv'
    counter = 1
    while os.path.exists(output_path):
        output_path = f'{OUTPUT_DIR}{filename}({counter})'
        counter += 1
    
    # Write output to file
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        title_row = ['img'] + [f'c{i}' for i in range(NUM_CLASSES)]
        writer.writerow(title_row, )
        for image_idx in range(len(image_names)):
            name = [image_names[image_idx]]
            data = list(map(str, y_predicted[image_idx]))
            row = name + data
            writer.writerow(row)

def save_model(model, model_name='model'):
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    with open(f'{MODEL_DIR}{model_name}.pkl', 'wb') as f:
        pickle.dump(model, f)

def load_model(filename):
    with open(f'{MODEL_DIR}{filename}.pkl', 'rb') as f:
        return pickle.load(f)

def _compress_image(source_path):
    new_path = f'compressed_{source_path}'
    
    if not os.path.exists(new_path):
        img = Image.open(source_path)
        new_size = tuple(s // RESIZE_IMAGE_RATIO for s in img.size)
        resized_img = img.resize(new_size)
        resized_img.save(new_path)

def compress_images():
    train_imgs_by_class = get_training_image_paths_by_class()
    test_imgs = _get_test_image_paths(_get_test_image_names())

    # Copy directories only
    if not os.path.exists(COMPRESSED_DATA_DIR):
        copytree(DATA_DIR, COMPRESSED_DATA_DIR, ignore=ignore_patterns('*.jpg', '*.csv'))
    
    # Copy and compress training images
    for class_i in range(len(train_imgs_by_class)):
        for path in train_imgs_by_class[class_i]:
            _compress_image(path)

    print('Training compressed.')

    # Copy and compress testing images
    for path in test_imgs:
        _compress_image(path)

    print('Compression complete.')

def create_driver_dictionary(filename):
    driver_dict = {}
    with open(filename) as f:
        reader = csv.reader(f)
        next(reader) # skip header
        for driver, c, img in reader:
            if driver not in driver_dict:
                driver_dict[driver] = []
            else:
                driver_dict[driver].append(os.path.join(TRAIN_DIR, c, img))
    return driver_dict

def generate_driver_split(driver_dict, num_test=3):
    test_paths = []
    train_paths = []
    test_labels = []
    train_labels = []
    for i, d in enumerate(driver_dict):
        path = driver_dict[d]
        if i < num_test:
            test_paths += path
            test_labels += [get_class_from_path(p) for p in path]
        else:
            train_paths += path
            train_labels += [get_class_from_path(p) for p in path]
    return train_paths, train_labels, test_paths, test_labels
