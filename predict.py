import sys
import cv2

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from functions import load_descriptions, prepare_images, add_predictions_to_images, add_text, generate_mosaic


def main():
    # Check command line arguments
    if len(sys.argv) not in [1, 2, 3]:
        sys.exit("Usage: python predict.py [model.keras] [data_directory]")
        # python predict.py model.keras gtsrb

    if len(sys.argv) == 1:
        model = "model.keras"
        directory = "gtsrb"
    elif len(sys.argv) == 2:
        model = sys.argv[1]
        directory = "gtsrb"
    else:
        model = sys.argv[1]
        directory = sys.argv[2]

    signs = load_descriptions("signs.csv")

    images = prepare_images(directory)
    image_arrays = [image['array'] for image in images]

    model = tf.keras.models.load_model(model)
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


if __name__ == "__main__":
    main()
