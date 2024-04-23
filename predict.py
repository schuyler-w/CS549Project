import sys

import numpy as np
import tensorflow as tf

from functions import load_descriptions, prepare_images, add_predictions_to_images, add_text, generate_mosaic


def main():
    # Check command line arguments
    if len(sys.argv) not in [1, 2, 3]:
        sys.exit("Usage: python predict.py [model.keras] [data_directory]")
        # python predict.py model.keras gtsrb

    model = sys.argv[1] if len(sys.argv) > 1 else "model.keras"
    directory = sys.argv[2] if len(sys.argv) == 3 else "gtsrb"

    signs = load_descriptions("signs.csv")

    images = prepare_images(directory)
    image_arrays = [image['array'] for image in images]

    model = tf.keras.models.load_model(model)
    predictions = model.predict(np.array(image_arrays))

    images = add_predictions_to_images(images, predictions)
    images = add_text(images, signs)

    generate_mosaic(images, signs)


if __name__ == "__main__":
    main()
