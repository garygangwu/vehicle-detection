import glob
import pickle
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.misc
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


output_fold = 'output_videos/'
x = 0
def save_fig(name, image):
  global x
  if '0_' in name:
    x += 1
  filename = output_fold + name + '_' + str(x) + '.jpg'
  scipy.misc.imsave(filename, image)


def process_image_v1(model_profile, image, debug = False):

  x_start_stop=[640, 1280]
  y_start_stop0, y_start_stop1, y_start_stop2, y_start_stop3, y_start_stop4= \
    [390, 650], [390, 650], [390, 550], [360, 500], [360, 470]
  windows_0 = slide_window(image, x_start_stop=[900, 1280], y_start_stop=y_start_stop0,
                        xy_window=(256, 256), xy_overlap=(0.9, 0.9))
  windows_1 = slide_window(image, x_start_stop=x_start_stop, y_start_stop=y_start_stop1,
                        xy_window=(144, 144), xy_overlap=(0.9, 0.9))
  windows_2 = slide_window(image, x_start_stop=x_start_stop, y_start_stop=y_start_stop2,
                        xy_window=(96, 96), xy_overlap=(0.9, 0.9))
  windows_3 = slide_window(image, x_start_stop=x_start_stop, y_start_stop=y_start_stop3,
                        xy_window=(72, 72), xy_overlap=(0.9, 0.9))
  windows_4 = slide_window(image, x_start_stop=[800, 1100], y_start_stop=y_start_stop3,
                        xy_window=(64, 64), xy_overlap=(0.9, 0.9))
  windows = windows_0 + windows_1 + windows_2 + windows_3 + windows_4

  hot_windows = search_windows(image, windows,
                               model_profile['model'],
                               model_profile['X_scaler'],
                               model_profile['params'])

  addition_weighted_windows = []
  for hot_window in hot_windows:
    if hot_window in windows_0:
      addition_weighted_windows.append(hot_window)
      addition_weighted_windows.append(hot_window)
      addition_weighted_windows.append(hot_window)
  hot_windows += addition_weighted_windows

  save_fig('0_original', image)
  save_fig('3_hot_windows', draw_boxes(image, hot_windows, color=(0, 255, 0), thick=2))

  heatmap = np.zeros_like(image[:, :, 0])
  heatmap = add_heat(heatmap, hot_windows)
  heatmap = apply_threshold(heatmap, 8)
  heatmap = np.clip(heatmap, 0, 255)

  labels = label(heatmap)
  draw_img = draw_labeled_bboxes(np.copy(image), labels)

  save_fig('4_heatmap', heatmap)
  save_fig('5_output', draw_img)

  if debug:
    return draw_img, heatmap, hot_windows
  return draw_img


def process_image_v2(model_profile, image, debug = False):
  y_start_stop = [390, 645] # Min and max in y to search in slide_window()
  hot_windows = find_car_windows(image, y_start_stop[0], y_start_stop[1],
                                 model_profile['model'],
                                 model_profile['X_scaler'],
                                 model_profile['params'])
  heatmap = np.zeros_like(image[:, :, 0])
  heatmap = add_heat(heatmap, hot_windows)
  heatmap = apply_threshold(heatmap, 2)
  heatmap = np.clip(heatmap, 0, 255)

  labels = label(heatmap)
  draw_img = draw_labeled_bboxes(np.copy(image), labels)
  if debug:
    return draw_img, heatmap, hot_windows
  return draw_img


def evaluate(model_profile, image):
  draw_img_1, heatmap_1, hot_windows_1 = process_image_v1(model_profile, image, debug = True)
  #draw_img_2, heatmap_2, hot_windows_2 = process_image_v2(model_profile, image, debug = True)

  fig = plt.figure(figsize=(18, 4))
  plt.subplot(131)
  plt.imshow(draw_img_1)
  plt.title('Final Output')
  plt.subplot(132)
  plt.imshow(heatmap_1, cmap='hot')
  plt.title('Heat Map')
  plt.subplot(133)
  plt.imshow(draw_boxes(image, hot_windows_1, color=(0, 0, 255), thick=6))
  plt.title('Windows')
  fig.tight_layout()
  plt.show()


def evaluate_all_models():
  model_profile = load_svc_models()['YCrCb']
  for filename in glob.glob('test_images/*.jpg'):
    image = mpimg.imread(filename)
    evaluate(model_profile, image)


def video(filename):
  clip = VideoFileClip(filename)
  model_profile = load_svc_models()['YCrCb']
  convert_image = lambda image: process_image_v1(model_profile, image)
  new_clip = clip.fl_image(convert_image)
  output_file = 'output_videos/' + filename
  new_clip.write_videofile(output_file, audio=False)


def image(filename, color):
  global output_fold
  output_fold = 'output_images/'
  model_profile = load_svc_models()[color]
  image = mpimg.imread(filename)
  print image.shape, image.dtype
  evaluate(model_profile, image)


debug_image_filenames = [
  'output_videos/0_original_717.jpg',
  'output_videos/0_original_719.jpg',
  'output_videos/0_original_722.jpg',
  'output_videos/0_original_725.jpg',
  'output_videos/0_original_728.jpg',
  'output_videos/0_original_731.jpg',
  'output_videos/0_original_168.jpg',
  'output_videos/0_original_156.jpg',
  'output_videos/0_original_183.jpg',
  'output_videos/0_original_746.jpg',
  'output_videos/0_original_682.jpg',
  'output_videos/0_original_888.jpg',
  'output_videos/0_original_1019.jpg',
  'output_videos/0_original_1150.jpg',
  'output_videos/0_original_1138.jpg',
  'output_videos/0_original_1224.jpg',
  'output_videos/0_original_1254.jpg',
  'output_videos/0_original_1253.jpg',
  'output_videos/0_original_1260.jpg',
  'output_videos/0_original_1261.jpg'
]

def main(mode, argv):
  if mode == 'evaluate':
    evaluate_all_models()
  elif mode == 'image':
    color = argv[0]
    filename = argv[1]
    image(filename, color)
  elif mode == 'images':
    color = argv[0]
    for filename in debug_image_filenames:
      print filename
      image(filename, color)
  elif mode == 'video':
    filename = argv[0]
    video(filename)
  else:
    print "Usage: python detect.py evaluate | image color filename | video filename | images"


if __name__ == "__main__":
  accept_modes = ['evaluate', 'image', 'video', 'images']
  if len(sys.argv) >= 2:
    mode = sys.argv[1]
  else:
    mode = 'evaluate'
  main(mode, sys.argv[2:])
