import os
import sys
import cv2
import csv
import random

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

IMG_OUTPUT_WIDTH = 180
IMG_OUTPUT_HEIGHT = 180
MOSAIC_LENGTH = 5
TEXT_POSITION_H = 5
TEXT_POSITION_V = -10


def main():
    # Check command line arguments
    if len(sys.argv) not in [3]:
        sys.exit("Usage: python predict.py model.keras test_directory")  # python predict.py model.keras gtsrb/Test

    signs = load_descriptions("signs.csv")

    images = prepare_images(sys.argv[2])
    image_arrays = [image['array'] for image in images]

    model = tf.keras.models.load_model(sys.argv[1])
    predictions = model.predict(np.array(image_arrays))

    images = add_predictions_to_images(images, predictions)
    images = add_text(images, signs)

    mosaic = generate_mosaic(images)
    mosaic = cv2.cvtColor(mosaic, cv2.COLOR_BGR2RGB)

    # Display the mosaic w/ matplotlib
    plt.imshow(mosaic)
    plt.title('Mosaic')
    plt.axis('off')

    plt.savefig('mosaic.png', bbox_inches='tight')
    plt.show()


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


def generate_mosaic(image_list):
    """
    Return a mosaic image of the input images

    :param image_list: list of images
    :return: mosaic image
    """

    final = []

    for i in range(0, len(image_list), MOSAIC_LENGTH):
        row = [image['complete'] for image in image_list[i:i + MOSAIC_LENGTH]]

        while len(row) < MOSAIC_LENGTH:
            row.append(np.zeros((180, 180, 3), np.uint8))

        final.append(cv2.hconcat(row))

    return cv2.vconcat(final)


if __name__ == "__main__":
    main()
