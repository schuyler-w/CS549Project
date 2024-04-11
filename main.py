import cv2
import csv
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.7 # Initial test size of 40%

def main():
    # Check command line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: ...")

    # get image arrays and labels for images
    images, labels = load_data(sys.argv[1])

def load_data(data_dir):
    """
    Load image data from directory "data_dir"

    """
    raise NotImplementedError

def get_model():
    """
    Return a compiled neural network model
    """
    raise NotImplementedError

if __name__ == "__main__":
    main()