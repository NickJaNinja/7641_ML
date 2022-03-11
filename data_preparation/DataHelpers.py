# Functions to assist in fetching data.

import os, os.path
import random
from PIL import Image
import numpy as np

DATA_DIR = 'data/'
TRAIN_DIR = DATA_DIR + 'train/'
TEST_DIR = DATA_DIR + 'test/'
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

def get_test_image_path():
    """Gets the path to all test images

    Returns:
        str: Path to the directory
    """
    return TEST_DIR

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
        list[list[int]]: Batched indices corresponding to images in training data.
    """
    num_images = get_num_training_images()
    image_paths = get_training_image_paths_flat()
    random.shuffle(image_paths)
    batch = np.linspace(0, num_images, num_batches+1, dtype=int)
    return [image_paths[batch[i]:batch[i+1]] for i in range(num_batches)]

def open_training_batch(batch):
    """Given a batch of paths, opens the whole batch of images and returns them.

    Returns:
        np.array[image]: Batch of images, shape (N x w x h x 3)
    """
    images = []
    for image_path in batch:
        img = open_image_as_np(image_path)
        images.append(img)
    images = np.array(images)
    return images
