import cv2
import csv
import numpy as np
import os
import sys
import tensorflow as tf

IMG_OUTPUT_WIDTH = 180
IMG_OUTPUT_HEIGHT = 180
MOSAIC_LENGTH = 15
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


def prepare_images(dir):
    """
    Load images from directory "dir" and return a list of images
    Resized to match training images.
    """
    images = []

    for file in os.listdir(dir):
        img = cv2.imread(os.path.join(dir, file))

        if img is not None:
            img = cv2.resize(img, (30, 30), interpolation=cv2.INTER_AREA)
            images.append(
                {'name': file,
                 'array': img}
            )

    return images


def add_preds_to_images(image_list, preds):
    """
    Adds prediction and confidence to each image dict

    :param image_list: image_list without predictions
    :param preds: input predictions
    :return: image_list with predictions and confidence added
    """

    for index, array in enumerate(preds):
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




def main():
    # Check command line arguments
    if len(sys.argv) not in [3]:
        sys.exit("Usage: python predict.py model.keras image_directory")

    signs = load_descriptions("signs.csv")
    images = prepare_images(sys.argv[2])
    image_arrays = [image['array'] for image in images]
    model = tf.keras.models.load_model(sys.argv[1])
    predictions = model.predict(np.array(image_arrays))
    images = add_preds_to_images(images, predictions)
    images = add_text(images, signs)
    mosaic = generate_mosaic(images)

    cv2.imshow('Mosaic', mosaic)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
