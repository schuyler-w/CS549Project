import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.5

def main():
    # Check command line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python main.py data_directory model.keras")

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

    # Predict the labels
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    report = classification_report(y_true_classes, y_pred_classes, target_names=[str(i) for i in range(NUM_CATEGORIES)])
    print(report)

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

        # Add 2 sequential 64 filter, 3x3 Convolutional Layers Followed by 2x2 Pooling
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Flatten layers
        tf.keras.layers.Flatten(),

        # Add A Dense Hidden layer with 512 units and 50% dropout
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dropout(0.5),

        # Add Dense Output layer with 43 output units
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])

    model.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    return model

if __name__ == "__main__":
    main()