# CS549 Project Traffic Identification

CS 549 - Machine Learning Final Project

Schuyler Wang, Alexander Pham, Thu Vu

___
This project aims to create a convolutional neural network to identify which traffic sign appears in a picture or photograph. 

The dataset used is the German Traffic Sign Recognition Benchmark (GTSRB) dataset, which contains 43 classes of traffic signs. The dataset is split into a training set and a test set, using `scikit-learn's` `train_test_split` function. The training set contains x% of the data, while the test set contains y% of the data. 

The images are 32x32 pixels in size and are in color. The dataset is available at https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign.

The results of the model are evaluated using the accuracy score, precision score, recall score, and F1 score. Furthermore the program returns an image mosaic of a random sample of qualifying images with their predicted labels overlaid using the `cv2` library.