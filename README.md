# CS549 Project Traffic Identification

CS 549 - Machine Learning Final Project

Schuyler Wang, Alexander Pham, Thu Vu

___
This project aims to create a convolutional neural network to identify which traffic sign appears in a picture or photograph. 

The dataset used is the German Traffic Sign Recognition Benchmark (GTSRB) dataset, which contains 43 classes of traffic signs. The dataset is split into a training set and a test set, using `scikit-learn's` `train_test_split` function. The training set contains x% of the data, while the test set contains y% of the data. 

The images are 32x32 pixels in size and are in color. The dataset is available at https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign.

The results of the model are evaluated using the accuracy score, precision score, recall score, and F1 score. Furthermore, the program returns an image mosaic of a random sample of qualifying images with their predicted labels overlaid using the `cv2` library.

___
## How to use

1. Clone the repository
2. Install the required libraries using `pip install -r requirements.txt`
3. Download the dataset from the link above

There are three main files in the repository:

main.py: This file contains the main code for the project. It reads the dataset, preprocesses the images, creates the model, trains the model, and evaluates the model. Returns a mosaic image of validation set images with predicted vs ground truth. Usage: `python main.py data_directory [filename.type]`. filename.type is optional, by default model will be saved to `model.keras`.



validation.py: This file performs cross-validation and benchmarking on a list of models to demonstrate how we selected the best model used in main.py. Usage: `python validation.py data_directory [output_filename]`. output_filename is optional, by default the output will be printed to standard output and not saved.

predict.py: This file contains the code to predict the traffic sign in an unlabeled testing set and returns a mosaic of predicted labels. Usage: `python predict.py model test_directory`. For example `python predict.py model.keras Test`. The program assumes that the test directory is inside the `gtsrb/` directory and automatically appends the file path. 