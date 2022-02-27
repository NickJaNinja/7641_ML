# Functions to assist in fetching data.

import os, os.path
import random
from PIL import Image
import numpy as np

DATA_DIR = 'data/'
TRAIN_DIR = DATA_DIR + 'train/'
TEST_DIR = DATA_DIR + 'test/'
NUM_CLASSES = 10


def get_training_image_path(driving_class: int = 0):
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

def get_test_image_path():
    """Gets the path to all test images

    Returns:
        str: Path to the directory
    """
    return TEST_DIR

def get_random_training_image_path():
    """Gets the path to a random training image in a random class

    Returns:
        str: Path to the image
    """
    random_class = random.randint(0, NUM_CLASSES-1)
    training_path = get_training_image_path(driving_class=random_class)
    imgs = [name for name in os.listdir(training_path)]

    num_images_in_class = len(imgs)
    random_image = imgs[random.randint(0, num_images_in_class-1)]
    path_to_image = f'{training_path}{random_image}'
    return path_to_image

def open_image_as_np(path_to_img: str):
    img = Image.open(path_to_img)
    img_as_np = np.asarray(img)
    return img_as_np