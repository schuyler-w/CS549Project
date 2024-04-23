import sys
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split

from functions import load_data

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.3


def main():
    # Check command-line arguments
    if len(sys.argv) not in [1, 2, 3]:
        sys.exit("Usage: python validation.py [data_directory] [benchmark.filetype]")
        # python validation.py gtsrb benchmark.txt
        # if you want to save the benchmark results to a file data_directory MUST be specified
        # otherwise prints results to standard out and benchmark is NOT saved

    if len(sys.argv) == 1:
        directory = "gtsrb"
    else:
        directory = sys.argv[1]

    # Get image arrays and labels for all image files
    images, labels = load_data(directory)

    # Convert to numpy arrays
    images, labels = np.array(images), np.array(labels)
    labels = tf.keras.utils.to_categorical(labels)

    # split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=TEST_SIZE)

    # Prepare models dictionary
    models = {
        "Model 1": tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
            tf.keras.layers.Dense(NUM_CATEGORIES,
                                  activation="softmax")
        ]),
        "Model 2": tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
            tf.keras.layers.Dense(32,
                                  activation="relu"),
            tf.keras.layers.Dense(NUM_CATEGORIES,
                                  activation="softmax")
        ]),

        "Model 3": tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3),
                                   activation="relu",
                                   input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64,
                                  activation="relu"),
            tf.keras.layers.Dense(NUM_CATEGORIES,
                                  activation="softmax")
        ]),

        "Model 4": tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(64, (3, 3),
                                   activation="relu",
                                   input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(64, (3, 3),
                                   activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128,
                                  activation="relu"),
            tf.keras.layers.Dense(NUM_CATEGORIES,
                                  activation="softmax")
        ]),

        "Model 5": tf.keras.models.Sequential([
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
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(NUM_CATEGORIES,
                                  activation="softmax")
        ]),

        "Model 6": tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(256, (3, 3),
                                   activation="relu",
                                   input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(256, (3, 3),
                                   activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512,
                                  activation="relu"),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(NUM_CATEGORIES,
                                  activation="softmax")
        ]),


        "Model 7": tf.keras.models.Sequential([
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
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(NUM_CATEGORIES,
                                  activation="softmax")
        ]),

        # Best model found
        "Model 8": tf.keras.models.Sequential([
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
    }

    # either output to standard out or to a file
    if len(sys.argv) <= 2:
        evaluate_models(models, x_train, x_test, y_train, y_test)
    else:
        with open(sys.argv[2], "w") as file:
            evaluate_models(models, x_train, x_test, y_train, y_test, stream=file)


def evaluate_models(models, x_train, x_test, y_train, y_test, stream=sys.stdout):
    original_stdout = sys.stdout  # Save the original stdout
    sys.stdout = stream  # Redirect output to the specified stream

    for name, model in models.items():
        print(f"Evaluating {name}")

        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        model.fit(x_train, y_train, epochs=EPOCHS, verbose=0)
        _, accuracy = model.evaluate(x_test, y_test, verbose=2)

        print(f"Accuracy for {name}: {accuracy}")

    sys.stdout = original_stdout

if __name__ == "__main__":
    main()
