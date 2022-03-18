# Functions to assist in fetching data.

import os, os.path
import csv
import random
from PIL import Image
import numpy as np

DATA_DIR = 'data/'
TRAIN_DIR = DATA_DIR + 'train/'
TEST_DIR = DATA_DIR + 'test/'
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

def _get_test_image_paths():
    """Gets the path to all test images

    Returns:
        str: Path to the directory
    """
    return [f'{TEST_DIR}{c}' for c in os.listdir(TEST_DIR)]

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

def open_training_batch(image_paths):
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
    test_image_paths = _get_test_image_paths()
    test_images = open_training_batch(test_image_paths)
    return (test_image_paths, test_images)

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
    output_path = f'{OUTPUT_DIR}{filename}'
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
