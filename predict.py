import cv2
import csv
import numpy as np
import os
import sys
import tensorflow as tf

def load_descriptions(csv_file):
    """
    Load image descriptions from csv file into a dictionary
    """

    descriptions = {}
    with open(csv_file) as f:
        reader = csv.reader(f)
        for row in reader:
            descriptions[row[0]] = row[1]

    return descriptions

def prepare_images(dir):
    """
    Load images from directory "dir" and return a list of images
    Resized to match training images.
    """

    images = []

    for file in os.listdir(dir):
        img = cv2.imread(os.path.join(dir, file))
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

    return image_list

def add_text(image_list, sign):
    """
    Adds text to each image in the list using cv2 library

    :param image_list: list of images
    :param sign: sign description
    :return: list of images with text added
    """

    return image_list

def return_mosaic(image_list):
    """
    Return a mosaic image of the input images

    :param image_list: list of images
    :return: mosaic image
    """

    return

def main():
    # Check command line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: ...")

if __name__ == "__main__":
    main()