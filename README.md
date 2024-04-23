# CS549 Project Traffic Identification

CS 549 - Machine Learning Final Project

Schuyler Wang, Alexander Pham, Thu Vu

___
This project aims to create a convolutional neural network to identify which traffic sign appears in a picture or photograph. 

The dataset used is the German Traffic Sign Recognition Benchmark (GTSRB) dataset, which contains 43 classes of traffic signs. The dataset is split into a training set and a test set, using `scikit-learn's` `train_test_split` function. The training set contains `70%` of the data, while the test set contains `30%` of the data.

The testing process also includes an unlabeled test set for manual confirmation. `predict.py` is used to predict the traffic sign in the unlabeled test set and returns a mosaic of the predicted labels. The program saves the mosaic image to `mosaic.png` in the current directory. 

The images are 32x32 pixels in size and are in color. The dataset is available at https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign.

The results of the model are evaluated using the accuracy score, precision score, recall score, and F1 score as well as the overall confusion matrix. Furthermore, the program returns an image mosaic of a random sample of qualifying images with their predicted labels overlaid with predictions and ground truth labels (if applicable) using `matplotlib`.

___
## How to use

1. Clone the repository
2. Install the required libraries using `pip install -r requirements.txt`
3. Download the dataset from the link above and place in program directory

There are three main files in the repository, each file has extra command line arguments that can be used to customize the program, although they are each designed with default values in mind so that they can be run without any arguments.

### main.py
This file contains the main code for the project. It reads the dataset, preprocesses the images, creates the model, trains the model, and evaluates the model.

Usage: `python main.py [data_directory] [filename.type] [metric_report.txt]`

Example: `python main.py gtsrb model.keras classification_report.txt`

Where data_directory is the directory where the dataset is stored, filename.type is the name of the file where the model will be saved, and classification_report.txt is the name of the file where the performance metrics will be saved.

`Data_directory`, `filename.type`, and `metric_report.txt` are optional, by default: program assumes the dataset is stored in the `gtsrb` directory, model will be saved to `model.keras`, and performance metrics are saved to `classification_report.txt`.

Further saves `prediction_results.png` and `confusion_matrix.png` to the current directory.

`prediction_results.png` is a mosaic image of validation set images with predicted vs ground truth.

![Prediction Results](/README_IMAGES/prediction_results.png)

`confusion_matrix.png` is a heatmap confusion matrix of the model's performance. Generated using `sklearn`'s `ConfusionMatrixDisplay` function.

![Confusion Matrix](/README_IMAGES/confusion_matrix.png)

### validation.py

This file performs cross-validation and benchmarking on a list of models to demonstrate how we selected the best model used in main.py. 

Usage: `python validation.py [data_directory] [benchmark.txt]`

Example: `python validation.py gtsrb benchmark.txt`

Where data_directory is the directory where the dataset is stored and benchmark.filetype is the file where the benchmarking results will be saved.

`Data_directory` and `benchmark.txt` are optional, by default the program assumes the `data_directory` is `gtsrb` and the output will be printed to standard output and not saved to a file.

### predict.py

This file contains the code to predict the traffic sign in an unlabeled testing set and returns a mosaic of predicted labels. 

Usage: `python predict.py [model] [data_directory]`

Example: `python predict.py model.keras gtsrb`

`model` and `data_directory` are optional, by default the program assumes the model is saved as `model.keras` in the current directory and the dataset is stored in the `gtsrb` directory.

The program assumes that the test directory is named `Test` and is inside the given directory and automatically appends the file path. ie: `./gtsrb/Test`

The program saves the mosaic image to `mosaic.png` in the current directory. Since the dataset is unlabeled, the program does not evaluate the model's performance and must be done so manually. This works for any dataset of images that are 32x32 pixels in size and fall inside one of the 43 categories.

![Mosaic](/README_IMAGES/mosaic.png)