# Vehicle Detection Project
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

<img src="demo/demo.gif" width="360" alt="demo" />

The goal of the project is to write a software pipeline to identify vehicles in a video from a front-facing camera on a car. It uses the Linear SVM classifier to train the classifer based on Histogram of Oriented Gradients (HOG), binned color, and color histograms as the training features, and implements a sliding-window technique to search for vehicles in video frames

---

# Implementation Deep Dive

## Model Training
The training code is located in [train.py](https://github.com/garygangwu/vehicle-detection/blob/master/train.py) and [feature_utils.py](https://github.com/garygangwu/vehicle-detection/blob/master/feature_utils.py). The labelled training data of [vehicles](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicles](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) is offered by Udacity from combination sources of the GTI vehicle image database and the KITTI vision benchmark suite. Here are examples of one of each of the `vehicle` and `non-vehicle` classes.

|Type|Sample Images for training|
|:---:|:---:|
|vehicles|<img src="demo/car1.png"/><img src="demo/car2.png"/><img src="demo/car3.png"/><img src="demo/car4.png"/><img src="demo/car5.png"/><img src="demo/car6.png"/><img src="demo/car7.png"/><img src="demo/car8.png"/><img src="demo/car9.png"/><img src="demo/car10.png"/>|
|not-vehicles|<img src="demo/noncar1.jpeg"/><img src="demo/noncar2.jpeg"/><img src="demo/noncar3.png"/><img src="demo/noncar4.jpeg"/><img src="demo/noncar5.png"/><img src="demo/noncar6.png"/><img src="demo/noncar7.png"/><img src="demo/noncar8.png"/><img src="demo/noncar9.png"/><img src="demo/noncar10.png"/>|

### Histogram of Oriented Gradients (HOG)

