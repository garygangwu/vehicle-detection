import numpy as np
import cv2
import glob
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from feature_utils import *

def read_image(image_file):
  if image_file.endswith('.png'):
    return (mpimg.imread(image_file) * 255).astype(np.uint8)
  return mpimg.imread(image_file)


def load_image_data():
  car_image_files = glob.glob('vehicles/**/*.png') + glob.glob('vehicles/**/*.jpg')
  notcar_image_files = glob.glob('non-vehicles/**/*.png') + glob.glob('non-vehicles/**/*.jpg')

  cars = []
  for image_file in car_image_files:
    img = read_image(image_file)
    assert img.dtype == np.uint8
    assert img.shape == (64,64,3)
    cars.append(img)
  notcars = []
  for image_file in notcar_image_files:
    img = read_image(image_file)
    assert img.dtype == np.uint8
    assert img.shape == (64,64,3)
    notcars.append(img)
  print "{} cars, and {} not-cars".format(len(cars), len(notcars))
  return cars, notcars


def get_training_params(color_space):
  return {
    'color_space': color_space, # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    'orient': 9,  # HOG orientations
    'pix_per_cell': 8, # HOG pixels per cell
    'cell_per_block': 2, # HOG cells per block
    'hog_channel': "ALL", # Can be 0, 1, 2, or "ALL"
    'spatial_size': (32, 32),
    'hist_bins': 32,    # Number of histogram bins
    'spatial_feat': True, # Spatial features on or off
    'hist_feat': True, # Histogram features on or off
    'hog_feat': True # HOG features on or off
  }


def get_training_features(car_images, notcar_images, training_params):
  car_features = extract_features(car_images,
                                  color_space=training_params['color_space'],
                                  spatial_size=training_params['spatial_size'],
                                  hist_bins=training_params['hist_bins'],
                                  orient=training_params['orient'],
                                  pix_per_cell=training_params['pix_per_cell'],
                                  cell_per_block=training_params['cell_per_block'],
                                  hog_channel=training_params['hog_channel'],
                                  spatial_feat=training_params['spatial_feat'],
                                  hist_feat=training_params['hist_feat'],
                                  hog_feat=training_params['hog_feat'])
  notcar_features = extract_features(notcar_images,
                                  color_space=training_params['color_space'],
                                  spatial_size=training_params['spatial_size'],
                                  hist_bins=training_params['hist_bins'],
                                  orient=training_params['orient'],
                                  pix_per_cell=training_params['pix_per_cell'],
                                  cell_per_block=training_params['cell_per_block'],
                                  hog_channel=training_params['hog_channel'],
                                  spatial_feat=training_params['spatial_feat'],
                                  hist_feat=training_params['hist_feat'],
                                  hog_feat=training_params['hog_feat'])
  print "{} car featuress, and {} non-car features".format(len(car_features), len(notcar_features))
  return car_features, notcar_features


def main():
  car_images, notcar_images = load_image_data()
  color_spaces = ['HSV', 'LUV', 'HLS', 'YUV', 'YCrCb']
  for color_space in color_spaces:
    params = get_training_params(color_space)
    car_features, notcar_features = get_training_features(car_images, notcar_images, params)
    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
      scaled_X, y, test_size=0.2, random_state=rand_state)

    svc = LinearSVC(C = 0.001)
    svc.fit(X_train, y_train)

    # Check the accuracy score of the SVC
    accuracy = svc.score(X_test, y_test)
    print "Test Accuracy of SVC = {} in {}".format(accuracy, color_space)

    # save them into a file
    store_model("svc_{}_model.p".format(color_space), svc, X_scaler, params)

if __name__ == "__main__":
  main()
