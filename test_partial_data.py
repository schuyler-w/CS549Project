import sys
import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from functions import load_partial_data, get_model, plot_classification_mosaic, plot_misclassified

EPOCHS = 10
NUM_CATEGORIES = 43
TEST_SIZE = 0.3

"""
This script trains a model on a smaller subset of the data and evaluates its performance.
It then generates a classification report and confusion matrix, and saves these to a file.
It also saves a confusion matrix plot and a plot of a 5x5 mosaic of random images from the test set.
Usage: python test_partial_data.py [proportion] [data_directory] [model_name.type] [metric_report.txt]
WARNING: Program may fail if the random sample of images does not contain all categories.
This may happen if the proportion of images is too low or just bad luck.
Just run again if it occurs.
"""

def main():
    # Check command line arguments
    if len(sys.argv) not in [1, 2, 3, 4, 5]:
        sys.exit("Usage: python test_partial_data.py [proportion] [data_directory] [model_name.type] [metric_report.txt]")

    # if no args
    if len(sys.argv) <= 2:
        directory = "gtsrb"
    else:
        directory = sys.argv[2]

    if len(sys.argv) > 1:
        prop = float(sys.argv[1])
    else:
        prop = 0.5

    # get image arrays and labels for images
    images, labels = load_partial_data(directory, prop)

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)

    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels),
        test_size=TEST_SIZE
    )

    # train model, function defined in functions.py
    model = get_model()

    model.fit(x_train, y_train,
              epochs=EPOCHS)

    model.evaluate(x_test, y_test, verbose=2)

    # Predict the labels
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    # Generate classification report and confusion matrix
    report = classification_report(y_true_classes, y_pred_classes, target_names=[str(i) for i in range(NUM_CATEGORIES)])
    print(report)

    cm = confusion_matrix(y_true_classes, y_pred_classes)
    print("Truncated Confusion Matrix:")
    print(cm)

    # set print options for numpy so that full confusion matrix is printed to report
    np.set_printoptions(threshold=np.inf, linewidth=200, edgeitems=10)

    # Write the classification report and confusion matrix to classification_report.txt
    if len(sys.argv) == 5:
        with open(sys.argv[4], 'w') as f:
            f.write("Partial Data Classification Report:\n")
            f.write(report + "\n")
            f.write("Confusion Matrix:\n")
            f.write(str(cm) + "\n")
        print(f"Report written to {sys.argv[4]}")
    else:
        with open("partial_data_classification_report.txt", 'w') as f:
            f.write("Partial Data Classification Report:\n")
            f.write(report + "\n")
            f.write("Confusion Matrix:\n")
            f.write(str(cm) + "\n")
        print("Report written to partial_data_classification_report.txt")

    # Display confusion matrix using ConfusionMatrixDisplay
    fig, ax = plt.subplots(figsize=(15, 15))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[str(i) for i in range(NUM_CATEGORIES)])
    disp.plot(include_values=True, cmap='viridis', ax=ax, xticks_rotation='vertical')
    plt.title('Confusion Matrix')
    plt.savefig('partial_data_confusion_matrix.png')
    print("Confusion matrix saved to partial_data_confusion_matrix.png")

    # Display a 5x5 plot of a set of random 25 images in the test set
    plot_classification_mosaic(x_test, y_true_classes, y_pred_classes, 'partial_data_prediction_results.png')
    print("Partial data prediction results saved to partial_data_prediction_results.png")

    # plot mosaic of misclassified images
    plot_misclassified(x_test, y_true_classes, y_pred_classes, 'partial_data_misclassified_results.png')
    print("Partial data misclassified results saved to partial_data_misclassified_results.png")

    # save model to file specified or default
    if len(sys.argv) >= 4:
        model.save(sys.argv[3])
        print(f"Model saved to {sys.argv[3]}.")
    else:
        model.save("partial_model.keras")
        print(f"Model saved to partial_model.keras.")


if __name__ == "__main__":
    main()
