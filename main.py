import os
import sys
import cv2

import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from helper import load_descriptions, load_data, get_model

from sklearn.metrics import classification_report

from predict import load_descriptions

EPOCHS = 10
IMG_WIDTH = 40
IMG_HEIGHT = 40
NUM_CATEGORIES = 43
TEST_SIZE = 0.5


def main():
    # Check command line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python main.py data_directory [filename.type]")

    # get image arrays and labels for images
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)

    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels),
        test_size=TEST_SIZE
    )

    model = get_model()

    model.fit(x_train, y_train,
              epochs=EPOCHS)

    model.evaluate(x_test, y_test, verbose=2)

    # Predict the labels
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    report = classification_report(y_true_classes, y_pred_classes, target_names=[str(i) for i in range(NUM_CATEGORIES)])
    print(report)

    # Display a 5x5 plot of a set of random 25 images in the test set
    indices = np.random.choice(np.arange(len(x_test)), 25, replace=False)
    sample_images = x_test[indices]
    sample_pred_classes = y_pred_classes[indices]
    sample_true_classes = y_true_classes[indices]

    descriptions = load_descriptions("signs.csv")

    plt.figure(figsize=(25, 25))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        prediction = descriptions[sample_pred_classes[i]]
        actual = descriptions[sample_true_classes[i]]
        col = 'g' if sample_pred_classes[i] == sample_true_classes[i] else 'r'
        plt.xlabel(f'Actual: {actual}\nPred: {prediction}', color=col)
        plt.imshow(sample_images[i])

    # Save the figure
    plt.savefig('prediction_results.png')  # Saves the plot as a PNG file

    if len(sys.argv) == 3:
        model.save(sys.argv[2])
        print(f"Model saved to {sys.argv[2]}.")
    else:
        model.save("model.keras")
        print(f"Model saved to model.keras.")

if __name__ == "__main__":
    main()
