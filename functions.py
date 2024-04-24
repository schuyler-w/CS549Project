import os
import cv2
import csv
import random
import sys

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43

IMG_OUTPUT_HEIGHT = 180
TEXT_POSITION_H = 5
TEXT_POSITION_V = -10


def load_descriptions(csv_file):
    """
    Load image descriptions from csv file into a dictionary

    :param csv_file: csv file containing image descriptions
    :return: dictionary of image descriptions
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

    :param directory: directory containing test images
    :return: list of images
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

    plt.figure(figsize=(n * 2, n * 2))
    plt.suptitle("Unlabeled Test Set Predictions", fontsize=16, color='black')
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
    :param data_dir: directory containing image data
        Load image data from directory "data_dir"
        data_dir has one directory named after each category, numbered 0 through NUM_CATEGORIES - 1.
        Inside each category directory are images of that category.
    :return (images, labels): tuple of lists of images and labels
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


def load_partial_data(data_dir, prop):
    """
    :param data_dir: directory containing image data
    :param prop: float, proportion of data to load (between 0 and 1)
        Load image data from directory "data_dir"
        data_dir has one directory named after each category, numbered 0 through NUM_CATEGORIES - 1.
        Inside each category directory are images of that category.
        Only a `prop` proportion of the data will be randomly loaded.
    :return (images, labels): tuple of lists of images and labels
    """
    if not (0 <= prop <= 1):
        sys.exit('Error: prop must be a value between 0 and 1.')

    images = []
    labels = []

    for i in range(NUM_CATEGORIES):
        directory = os.path.join(data_dir, "Train", str(i))
        files = os.listdir(directory)
        num_files_to_select = int(len(files) * prop)
        selected_files = random.sample(files, num_files_to_select)  # Randomly select a proportion of files

        for file in selected_files:
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
    :return: compiled neural network model
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


def plot_classification_mosaic(x_test, y_true_classes, y_pred_classes, save_path):
    """
    Plots a 5x5 mosaic of random 25 images from the test set showing ground truth vs predictions.

    Parameters:
    x_test (np.array): The array of test images.
    y_true_classes (np.array): The array of true class indices for the test images.
    y_pred_classes (np.array): The array of predicted class indices for the test images.
    save_path (str): Path to save the plot image.
    """
    # Randomly select 25 indices
    indices = np.random.choice(np.arange(len(x_test)), 25, replace=False)
    sample_images = x_test[indices]
    sample_pred_classes = y_pred_classes[indices]
    sample_true_classes = y_true_classes[indices]

    # Load descriptions
    descriptions = load_descriptions("signs.csv")

    # Plot mosaic of ground truth vs predictions
    plt.figure(figsize=(10, 10))
    plt.suptitle("Validation: Ground Truth vs Prediction", fontsize=16, color='black')
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        prediction = descriptions[sample_pred_classes[i]]
        actual = descriptions[sample_true_classes[i]]
        col = 'g' if sample_pred_classes[i] == sample_true_classes[i] else 'r'
        plt.xlabel(f'True: {actual}\nPred: {prediction}', color=col, fontsize=6)
        plt.imshow(sample_images[i])

    plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1, wspace=0.4, hspace=0.4)
    plt.savefig(save_path)


def plot_misclassified(x_test, y_true_classes, y_pred_classes, save_path):
    """
    Plots 25 randomly selected misclassified images.

    Parameters:
    x_test (np.array): The array of test images.
    y_true_classes (np.array): The array of true class indices for the test images.
    y_pred_classes (np.array): The array of predicted class indices for the test images.\
    save_path (str): Path to save the plot image.
    """
    # Identify misclassified images
    misclassified_indices = np.where(y_pred_classes != y_true_classes)[0]

    # Ensure there are at least 25 misclassified images to sample from
    if len(misclassified_indices) >= 25:
        sample_indices = np.random.choice(misclassified_indices, 25, replace=False)
    else:
        sample_indices = misclassified_indices  # Use all if less than 25

    # Get the misclassified images and labels for the sample
    sample_images = x_test[sample_indices]
    sample_pred_classes = y_pred_classes[sample_indices]
    sample_true_classes = y_true_classes[sample_indices]

    # Load descriptions
    descriptions = load_descriptions("signs.csv")

    # Plot misclassified images
    plt.figure(figsize=(10, 10))
    plt.suptitle("Misclassified Images: Ground Truth vs Prediction", fontsize=16, color='black')
    for i in range(len(sample_indices)):
        plt.subplot(5, 5, i + 1)
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        prediction = descriptions[sample_pred_classes[i]]
        actual = descriptions[sample_true_classes[i]]
        plt.xlabel(f'True: {actual}\nPred: {prediction}', color='r', fontsize=6)
        plt.imshow(sample_images[i])

    plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1, wspace=0.4, hspace=0.4)
    plt.savefig(save_path)


def plot_unlabeled(model, data_dir):
    """
    Plot a 5x5 mosaic of random 25 images from the test set with predictions and confidence.

    Parameters:
    model (tf.keras.Model): The trained model.
    data_dir (str): The directory containing the test images.
    """
    # Load descriptions
    descriptions = load_descriptions("signs.csv")
    model = tf.keras.models.load_model(model)

    # Prepare images
    images = prepare_images(data_dir)
    image_arrays = [image['array'] for image in images]

    # Predict
    predictions = model.predict(np.array(image_arrays))

    # Add predictions and confidence to images
    images = add_predictions_to_images(images, predictions)
    images = add_text(images, descriptions)

    # Generate mosaic
    generate_mosaic(images, descriptions)
