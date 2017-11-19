import glob
import pickle
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip
from feature_utils import *
from car_detection_utils import *

def load_svc_models():
  model_file_names = glob.glob('svc_*_model.p')
  svc_model_profiles = {}
  for file_name in model_file_names:
    svc, X_scaler, training_params = pickle.load(open(file_name, 'rb'))
    svc_model_profiles[ training_params['color_space'] ] = {
      'model': svc,
      'X_scaler': X_scaler,
      'params': training_params
    }
  return svc_model_profiles


def process_image_v1(model_profiles, image, debug = False):

  y_start_stop1, y_start_stop2, y_start_stop3 = [400, 650], [400, 600], [400, 550]
  windows_1 = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop1,
                        xy_window=(128, 128), xy_overlap=(0.75, 0.75))
  windows_2 = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop2,
                        xy_window=(96, 96), xy_overlap=(0.75, 0.75))
  windows_3 = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop3,
                        xy_window=(64, 64), xy_overlap=(0.75, 0.75))
  windows = windows_1 + windows_2 + windows_3

  hot_windows = list(windows) # make a copy
  for model_profile in model_profiles:
    hot_window_candidates = search_windows(image, windows,
                                           model_profile['model'],
                                           model_profile['X_scaler'],
                                           model_profile['params'])
    hot_windows = list(set(hot_windows) & set(hot_window_candidates))

  heatmap = np.zeros_like(image[:, :, 0])
  heatmap = add_heat(heatmap, hot_windows)
  heatmap = apply_threshold(heatmap, 2)
  heatmap = np.clip(heatmap, 0, 255)

  labels = label(heatmap)
  draw_img = draw_labeled_bboxes(np.copy(image), labels)

  if debug:
    return draw_img, heatmap, hot_windows
  return draw_img


def process_image_v2(model_profiles, image, debug = False):
  y_start_stop = [390, 645] # Min and max in y to search in slide_window()
  hot_windows = None
  for model_profile in model_profiles:
    hot_window_candidates = find_car_windows(image, y_start_stop[0], y_start_stop[1],
                                             model_profile['model'],
                                             model_profile['X_scaler'],
                                             model_profile['params'])
    if hot_windows == None:
      hot_windows = hot_window_candidates
    else:
      hot_windows = list(set(hot_windows) & set(hot_window_candidates))
  heatmap = np.zeros_like(image[:, :, 0])
  heatmap = add_heat(heatmap, hot_windows)
  heatmap = apply_threshold(heatmap, 2)
  heatmap = np.clip(heatmap, 0, 255)

  labels = label(heatmap)
  draw_img = draw_labeled_bboxes(np.copy(image), labels)
  if debug:
    return draw_img, heatmap, hot_windows
  return draw_img


def evaluate(model_profiles, image):
  draw_img_1, heatmap_1, hot_windows_1 = process_image_v1(model_profiles, image, debug = True)
  draw_img_2, heatmap_2, hot_windows_2 = process_image_v2(model_profiles, image, debug = True)

  fig = plt.figure()
  plt.subplot(331)
  plt.imshow(draw_img_1)
  plt.title('v1')
  plt.subplot(332)
  plt.imshow(heatmap_1, cmap='hot')
  plt.title('Heat Map v1')
  plt.subplot(333)
  plt.imshow(draw_boxes(image, hot_windows_1))
  plt.title('boxes v1')
  plt.subplot(334)
  plt.imshow(draw_img_2)
  plt.title('v2')
  fig.tight_layout()
  plt.subplot(335)
  plt.imshow(heatmap_2, cmap='hot')
  plt.title('Heat Map v2')
  plt.subplot(336)
  plt.imshow(draw_boxes(image, hot_windows_2))
  plt.title('boxes')
  fig.tight_layout()
  plt.show()


def evaluate_all_models():
  model_profiles = load_svc_models()
  for filename in glob.glob('test_images/*.jpg'):
    image = mpimg.imread(filename)
    for color_space, model_profile in model_profiles.iteritems():
      evaluate(model_profile, image)


def video(filename):
  clip = VideoFileClip(filename)
  model_profiles = [ load_svc_models()['YCrCb'] ] # load_svc_models()['YUV']
  convert_image = lambda image: process_image_v1(model_profiles, image)
  new_clip = clip.fl_image(convert_image)
  output_file = 'output_videos/' + filename
  new_clip.write_videofile(output_file, audio=False)


def image(filename, color):
  model_profiles = [ load_svc_models()[color] ]
  image = mpimg.imread(filename)
  evaluate(model_profiles, image)


def main(mode, argv):
  if mode == 'evaluate':
    evaluate_all_models()
  elif mode == 'image':
    color = argv[0]
    filename = argv[1]
    image(filename, color)
  elif mode == 'video':
    filename = argv[0]
    video(filename)
  else:
    print "Usage: python detect.py evaluate | image color filename | video filename"


if __name__ == "__main__":
  accept_modes = ['evaluate', 'image', 'video']
  mode = 'evaluate'
  if len(sys.argv) >= 2:
    mode = sys.argv[1]
    if not mode in accept_modes:
      mode = 'evaluate'
  main(mode, sys.argv[2:])
