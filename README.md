# Vehicle Detection Project
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

<img src="demo/demo.gif" width="360" alt="original Image" />

The goal of the project is to write a software pipeline to identify vehicles in a video from a front-facing camera on a car. It uses the Linear SVM classifier to train the classifer based on Histogram of Oriented Gradients (HOG), binned color, and color histograms as the training features, and implements a sliding-window technique to search for vehicles in video frames

---

# Implementation Deep Dive

## Model Training
The training code is located in [train.py](https://github.com/garygangwu/vehicle-detection/blob/master/train.py) and [feature_utils.py](https://github.com/garygangwu/vehicle-detection/blob/master/feature_utils.py). The labelled training data of [vehicles](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicles](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) is offered by Udacity from combination sources of the GTI vehicle image database and the KITTI vision benchmark suite.

### Histogram of Oriented Gradients (HOG)

