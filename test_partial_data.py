import sys
import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from functions import load_descriptions, load_partial_data, get_model

EPOCHS = 10
NUM_CATEGORIES = 43
TEST_SIZE = 0.3

"""
This script trains a model on a smaller subset of the data and evaluates its performance.
It then generates a classification report and confusion matrix, and saves these to a file.
It also saves a confusion matrix plot and a plot of a 5x5 mosaic of random images from the test set.
Usage: python main.py [data_directory] [model_name.type] [metric_report.txt] [proportion]
"""

def main():
    # Check command line arguments
    if len(sys.argv) not in [1, 2, 3, 4, 5]:
        sys.exit("Usage: python main.py [data_directory] [model_name.type] [metric_report.txt] [proportion]")

    # if no args
    if len(sys.argv) == 1:
        directory = "gtsrb"
    else:
        directory = sys.argv[1]

    if len(sys.argv) == 5:
        prop = sys.argv[4]
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
    if len(sys.argv) == 4:
        with open(sys.argv[3], 'w') as f:
            f.write("Partial Data Classification Report:\n")
            f.write(report + "\n")
            f.write("Confusion Matrix:\n")
            f.write(str(cm) + "\n")
        print(f"Report written to {sys.argv[3]}")
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
    indices = np.random.choice(np.arange(len(x_test)), 25, replace=False)
    sample_images = x_test[indices]
    sample_pred_classes = y_pred_classes[indices]
    sample_true_classes = y_true_classes[indices]

    # map numeric labels -> descriptions of signs
    descriptions = load_descriptions("signs.csv")

    # plot mosaic of ground truth vs preds
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

    # Save the figure
    plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1, wspace=0.4, hspace=0.4)
    plt.savefig('partial_data_prediction_results.png')  # Saves the plot as a PNG file

    # save model to file specified or default
    if len(sys.argv) == 3:
        model.save(sys.argv[2])
        print(f"Model saved to {sys.argv[2]}.")
    else:
        model.save("partial_model.keras")
        print(f"Model saved to partial_model.keras.")


if __name__ == "__main__":
    main()
