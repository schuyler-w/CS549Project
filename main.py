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
TEST_SIZE = 0.7 # Initial test size of 70%

def main():
    # Check command line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # get image arrays and labels for images
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    model = get_model()

    model.fit(x_train, y_train, epochs=EPOCHS)

    model.evaluate(x_test, y_test, verbose=2)

    if len(sys.argv) == 3:
        model.save(sys.argv[2])
        print(f"Model saved to {sys.argv[2]}.")


def load_data(data_dir):
    """
    Load image data from directory "data_dir"

    data_dir has one directory named after each category, numbered 0 through NUM_CATEGORIES - 1.
    Inside each category directory are images of that category.

    Returns a tuple (images, labels) where images is a list of images and labels is a list of labels.

    """
    images = []
    labels = []

    for i in range(NUM_CATEGORIES):
        directory = os.path.join(data_dir, "Train", str(i))
        for file in os.listdir(directory):
            img = cv2.imread(os.path.join(directory, file))
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
            images.append(img)
            labels.append(i)

    return images, labels

def get_model():
    """
    Return a compiled neural network model
    """
    model = tf.keras.models.Sequential([

    ])

if __name__ == "__main__":
    main()