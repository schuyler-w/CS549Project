import cv2
import numpy as np
import os
import sys
import tensorflow as tf
from sklearn.model_selection import train_test_split, LeaveOneOut
from main import load_data

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.7

def main():
    # Check command-line arguments
    if len(sys.argv) not in [2]:
        sys.exit("Usage: python validation.py data_directory")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Prepare models
    models = {
        "Model 1": tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
            tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
        ]),
        "Model 2": tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
        ]),
        "Model 3": tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
        ]),
        "Model 4": tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
        ])
    }

    # Convert labels to categorical
    labels = tf.keras.utils.to_categorical(labels)

    # LOOCV Setup
    loo = LeaveOneOut()
    images, labels = np.array(images), np.array(labels)

    # Evaluate each model using LOOCV
    for name, model in models.items():
        print(f"Evaluating {name}")
        accuracies = []
        for train_index, test_index in loo.split(images):
            x_train, x_test = images[train_index], images[test_index]
            y_train, y_test = labels[train_index], labels[test_index]

            model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
            model.fit(x_train, y_train, epochs=EPOCHS, verbose=0)
            _, accuracy = model.evaluate(x_test, y_test, verbose=0)
            accuracies.append(accuracy)

        print(f"Average accuracy for {name}: {np.mean(accuracies)}")


if __name__ == "__main__":
    main()
