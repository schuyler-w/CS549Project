import os
import cv2
import csv
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import sys

IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.3

IMG_OUTPUT_WIDTH = 180
IMG_OUTPUT_HEIGHT = 180
MOSAIC_LENGTH = 5
TEXT_POSITION_H = 5
TEXT_POSITION_V = -10


def load_descriptions(csv_file):
    """
    Load image descriptions from csv file into a dictionary
    """

    descriptions = {}

    with open(csv_file) as f:
        reader = csv.reader(f)
        reader_row = next(reader)
        for row in reader:
            descriptions[int(row[0])] = row[1]

    return descriptions


def prepare_images(directory):
    """
    Load images from test directory "dir" and return a list of 500 random images
    Resized to match training images.
    """
    images = []

    directory = os.path.join(directory, 'Test')

    all_files = os.listdir(directory)
    selected_files = random.sample(all_files, min(25, len(all_files)))

    for file in selected_files:
        img = cv2.imread(os.path.join(directory, file))

        if img is not None:
            img = cv2.resize(img, (30, 30), interpolation=cv2.INTER_AREA)
            images.append(
                {'name': file,
                 'array': img}
            )

    return images


def add_predictions_to_images(image_list, predictions):
    """
    Adds prediction and confidence to each image dict

    :param image_list: image_list without predictions
    :param predictions: input predictions
    :return: image_list with predictions and confidence added
    """

    for index, array in enumerate(predictions):
        prediction = np.argmax(array)

        confidence = "{:.0%}".format(array[prediction])
        image_list[index]['prediction'] = prediction
        image_list[index]['confidence'] = confidence

    return image_list


def add_text(image_list, sign):
    """
    Adds text to each image in the list using cv2 library

    :param image_list: list of images
    :param sign: sign description
    :return: list of images with text added
    """
    for image in image_list:
        image['complete'] = cv2.putText(
            cv2.resize(image['array'], (180, 180)),
            sign[image['prediction']].upper(),
            (TEXT_POSITION_H, IMG_OUTPUT_HEIGHT + TEXT_POSITION_V),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 180, 0),
            thickness=1
        )

    return image_list


def generate_mosaic(image_list, descriptions):
    """
    Return a mosaic image of the input images

    :param descriptions: mapping of sign numeric label to description
    :param image_list: list of images
    :return: mosaic image
    """

    n = int(len(image_list) ** 0.5)
    n = max(n, 5)  # Ensuring at least a 5x5 grid

    plt.figure(figsize=(n * 2, n * 2))  # Adjust the figure size as needed
    for i, image in enumerate(image_list):
        plt.subplot(n, n, i + 1)
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])

        prediction = image['prediction']
        predicted_text = descriptions[prediction].upper()

        plt.xlabel(f'{predicted_text}', color='green', fontsize=8)

        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(cv2.resize(image['array'], (180, 180)), cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb)

    plt.tight_layout()  # Adjust subplots to give more room for labels
    plt.savefig('mosaic.png')  # Save the complete mosaic
    plt.show()


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

    if len(images) != len(labels):
        sys.exit('Error when loading data, number of images did not match number of labels.')
    else:
        print(f'{len(images)}, {len(labels)} labelled images loaded successfully from dataset!')

    return images, labels


def get_model():
    """
    Return a compiled neural network model
    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(128, (3, 3),
                               activation="relu",
                               input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Conv2D(128, (3, 3),
                               activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256,
                              activation="relu"),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(NUM_CATEGORIES,
                              activation="softmax")
    ])

    model.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    return model
