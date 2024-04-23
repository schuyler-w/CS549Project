import sys
import cv2

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from helper import load_descriptions, prepare_images, add_predictions_to_images, add_text, generate_mosaic

IMG_OUTPUT_WIDTH = 180
IMG_OUTPUT_HEIGHT = 180
MOSAIC_LENGTH = 5
TEXT_POSITION_H = 5
TEXT_POSITION_V = -10


def main():
    # Check command line arguments
    if len(sys.argv) not in [3]:
        sys.exit("Usage: python predict.py model.keras test_directory")  # python predict.py model.keras gtsrb/Test

    signs = load_descriptions("signs.csv")

    images = prepare_images(sys.argv[2])
    image_arrays = [image['array'] for image in images]

    model = tf.keras.models.load_model(sys.argv[1])
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
