import numpy as np
import cv2
from skimage.feature import hog
import pickle

def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                     vis=False, feature_vec=True):
  # Call with two outputs if vis==True
  if vis == True:
    features, hog_image = hog(img, orientations=orient,
			      pixels_per_cell=(pix_per_cell, pix_per_cell),
			      cells_per_block=(cell_per_block, cell_per_block),
			      transform_sqrt=True,
			      visualise=vis, feature_vector=feature_vec,
			      block_norm='L2-Hys')
    return features, hog_image
  # Otherwise call with one output
  else:
    features = hog(img, orientations=orient,
		   pixels_per_cell=(pix_per_cell, pix_per_cell),
		   cells_per_block=(cell_per_block, cell_per_block),
		   transform_sqrt=True,
		   visualise=vis, feature_vector=feature_vec,
		   block_norm='L2-Hys')
    return features


def bin_spatial(img, size=(32, 32)):
  # Use cv2.resize().ravel() to create the feature vector
  features = cv2.resize(img, size).ravel()
  # Return the feature vector
  return features


def color_hist(img, nbins=32, bins_range=(0, 256)):
  # Compute the histogram of the color channels separately
  channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
  channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
  channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
  # Concatenate the histograms into a single feature vector
  hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
  # Return the individual histograms, bin_centers and feature vector
  return hist_features


def convert_color(image, color_space='YCrCb'):
  if color_space == 'HSV':
    return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
  if color_space == 'LUV':
    return cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
  if color_space == 'HLS':
    return cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
  if color_space == 'YUV':
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
  if color_space == 'YCrCb':
     return cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
  print "return unchanged color_space {}".format(color_space)
  return np.copy(image)


def single_img_features(img, params):
  img_features = []
  feature_image = np.copy(img)
  if params['spatial_feat'] == True:
    spatial_features = bin_spatial(feature_image, size=params['spatial_size'])
    img_features.append(spatial_features)
  if params['hist_feat'] == True:
    hist_features = color_hist(feature_image, nbins=params['hist_bins'])
    img_features.append(hist_features)
  if params['hog_feat'] == True:
    if params['hog_channel'] == 'ALL':
      hog_features = []
      for channel in range(feature_image.shape[2]):
        hog_features.extend(
          get_hog_features(feature_image[:,:,channel],
                           params['orient'], params['pix_per_cell'], params['cell_per_block'],
                           vis=False, feature_vec=True))
    else:
      hog_features = get_hog_features(feature_image[:,:,params['hog_channel']],
                                      params['orient'], params['pix_per_cell'], params['cell_per_block'],
                                      vis=False, feature_vec=True)
    img_features.append(hog_features)
  return np.concatenate(img_features)


# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(images, color_space='RGB', spatial_size=(32, 32),
                     hist_bins=32, orient=9,
                     pix_per_cell=8, cell_per_block=2, hog_channel=0,
                     spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for image in images:
      file_features = []
      # apply color conversion if other than 'RGB'
      feature_image = convert_color(image, color_space)

      if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        file_features.append(spatial_features)
      if hist_feat == True:
        # Apply color_hist()
        hist_features = color_hist(feature_image, nbins=hist_bins)
        file_features.append(hist_features)
      if hog_feat == True:
        if hog_channel == 'ALL':
          hog_features = []
          for channel in range(feature_image.shape[2]):
            hog_features.append(
              get_hog_features(feature_image[:,:,channel],
                               orient, pix_per_cell, cell_per_block,
                               vis=False, feature_vec=True))
          hog_features = np.ravel(hog_features)
        else:
          hog_features = get_hog_features(
            feature_image[:,:,hog_channel], orient, pix_per_cell, cell_per_block,
            vis=False, feature_vec=True)
        # Append the new feature vector to the features list
        file_features.append(hog_features)
      features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features


def store_model(file_name, svc, X_scaler, training_params):
  with open(file_name, "wb") as f:
    pickle.dump((svc, X_scaler, training_params), f)
  print "saved the model to {}".format(file_name)
