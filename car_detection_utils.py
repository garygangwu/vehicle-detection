import numpy as np
import cv2
from feature_utils import *

PREDICTION_THRESH = 1.

# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
  # Make a copy of the image
  imcopy = np.copy(img)
  # Iterate through the bounding boxes
  for bbox in bboxes:
    # Draw a rectangle given bbox coordinates
    cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
  # Return the image copy with boxes drawn
  return imcopy


def add_heat(heatmap, bbox_list):
  # Iterate through list of bboxes
  for box in bbox_list:
    # Add += 1 for all pixels inside each bbox
    # Assuming each "box" takes the form ((x1, y1), (x2, y2))
    heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
  # Return updated heatmap
  return heatmap


def apply_threshold(heatmap, threshold):
  # Zero out pixels below the threshold
  heatmap[heatmap <= threshold] = 0
  # Return thresholded map
  return heatmap


def draw_labeled_bboxes(img, labels):
  # Iterate through all detected cars
  for car_number in range(1, labels[1]+1):
    # Find pixels with each car_number label value
    nonzero = (labels[0] == car_number).nonzero()
    # Identify x and y values of those pixels
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Define a bounding box based on min/max x and y
    bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
    # Draw the box on the image
    cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
  # Return the image
  return img


def model_prediction(svc, test_features):
  scores = svc.decision_function(test_features)
  prediction = svc.predict(test_features)
  if prediction == 1:
    if scores[0] > PREDICTION_THRESH:
      #print scores
      return 1
  return 0


def find_car_windows(img, ystart, ystop, svc, X_scaler, params):
  on_windows = []

  img_tosearch = img[ystart:ystop,:,:]
  ctrans_tosearch = convert_color(img_tosearch, color_space=params['color_space'])
  ctrans_tosearch = ctrans_tosearch.astype(np.float32)

  ch1 = ctrans_tosearch[:,:,0]
  ch2 = ctrans_tosearch[:,:,1]
  ch3 = ctrans_tosearch[:,:,2]

  # Define blocks and steps as above
  nxblocks = (ch1.shape[1] // params['pix_per_cell']) - params['cell_per_block'] + 1
  nyblocks = (ch1.shape[0] // params['pix_per_cell']) - params['cell_per_block'] + 1
  nfeat_per_block = params['orient']*params['cell_per_block']**2

  # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
  window = 64
  nblocks_per_window = (window // params['pix_per_cell']) - params['cell_per_block'] + 1
  cells_per_step = 2  # Instead of overlap, define how many cells to step
  nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
  nysteps = (nyblocks - nblocks_per_window) // cells_per_step

  # Compute individual channel HOG features for the entire image
  hog1 = get_hog_features(ch1, params['orient'], params['pix_per_cell'], params['cell_per_block'], vis=False, feature_vec=False)
  hog2 = get_hog_features(ch2, params['orient'], params['pix_per_cell'], params['cell_per_block'], vis=False, feature_vec=False)
  hog3 = get_hog_features(ch3, params['orient'], params['pix_per_cell'], params['cell_per_block'], vis=False, feature_vec=False)

  for xb in range(nxsteps):
    for yb in range(nysteps):
      ypos = yb*cells_per_step
      xpos = xb*cells_per_step
      # Extract HOG for this patch
      hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
      hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
      hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
      hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

      xleft = xpos*params['pix_per_cell']
      ytop = ypos*params['pix_per_cell']

      # Extract the image patch
      subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))

      # Get color features
      spatial_features = bin_spatial(subimg, size=params['spatial_size'])
      hist_features = color_hist(subimg, nbins=params['hist_bins'])

      # Scale features and make a prediction
      features = np.hstack((spatial_features, hist_features, hog_features))
      test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
      test_prediction = model_prediction(svc, test_features)

      if test_prediction == 1:
        xbox_left = np.int(xleft)
        ytop_draw = np.int(ytop)
        win_draw = np.int(window)
        on_windows.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))
  return on_windows


# Define a function that takes an image,
# start and stop positions in both x and y,
# window size (x and y dimensions),
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step)
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step)
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
      for xs in range(nx_windows):
        # Calculate window position
        startx = xs*nx_pix_per_step + x_start_stop[0]
        endx = startx + xy_window[0]
        starty = ys*ny_pix_per_step + y_start_stop[0]
        endy = starty + xy_window[1]

        # Append window position to list
        window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list


# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, svc, X_scaler, params):
  on_windows = []
  converted_img = convert_color(img, color_space=params['color_space'])
  for window in windows:
    test_img = cv2.resize(
        converted_img[window[0][1]:window[1][1], window[0][0]:window[1][0]],
        (64, 64))
    features = single_img_features(test_img, params)
    test_features = X_scaler.transform(features).reshape(1,-1)
    prediction = model_prediction(svc, test_features)
    if prediction == 1:
      #print window
      on_windows.append(window)

  return on_windows
