
# coding: utf-8

# # Object Detection Demo
# Welcome to the object detection inference walkthrough!  This notebook will walk you step by step through the process of using a pre-trained model to detect objects in an image. Make sure to follow the [installation instructions](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/installation.md) before you start.

# # Imports

# In[7]:

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import copy
import config

from collections import defaultdict
from io import StringIO

os.environ['QT_QPA_PLATFORM']='offscreen' # add
#from matplotlib import pyplot as plt
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from PIL import Image

import argparse
import re

exts = config.exts
# arg
parser = argparse.ArgumentParser()
parser.add_argument(
  '--target_dir',
  type=str,
  default='',
  help='image files target directry.'
)
FLAGS, unparsed = parser.parse_known_args()
# ## Env setup

# In[8]:

# This is needed to display the images.
#get_ipython().magic('matplotlib inline')

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")


# ## Object detection imports
# Here are the imports from the object detection module.

# In[9]:

from utils import label_map_util

from utils import visualization_utils as vis_util

# # Model preparation 

# ## Variables
# 
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.  
# 
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.

# In[ ]:

# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_PATH = '/datadrive/ssd/'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

pass_accuracy_rate = 60

# ## Download Model

# In[ ]:

opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_PATH + MODEL_FILE)
tar_file = tarfile.open(MODEL_PATH + MODEL_FILE)
for file in tar_file.getmembers():
  file_name = os.path.basename(file.name)
  if 'frozen_inference_graph.pb' in file_name:
    tar_file.extract(file, os.getcwd())


# ## Load a (frozen) Tensorflow model into memory.

# In[ ]:

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

# In[ ]:

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# ## Helper code

# In[ ]:

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  print('[image size]{} / {} : {}'.format(im_width, im_height, image.getdata()))
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


# # Detection

# In[ ]:

# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = '{}/{}'.format(config.TEST_IMAGES_DIR, FLAGS.target_dir)
#TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3) ]
PATH_TO_OUTPUT_IMAGES_DIR = '{}/{}'.format(config.OUTPUT_IMAGES_DIR, FLAGS.target_dir)
#OUTPUT_IMAGE_PATHS = [ os.path.join(PATH_TO_OUTPUT_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3) ]
PATH_TO_CROP_IMAGES_DIR = '{}/{}'.format(config.CROP_IMAGES_DIR, FLAGS.target_dir)
# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

if not os.path.isdir(PATH_TO_CROP_IMAGES_DIR):
  os.makedirs(PATH_TO_CROP_IMAGES_DIR)
#for folder in [PATH_TO_OUTPUT_IMAGES_DIR, PATH_TO_CROP_IMAGES_DIR]:
#  if not os.path.isdir(folder):
#    os.makedirs(folder)

# In[ ]:

# メモリ確保
config = tf.ConfigProto(
  gpu_options=tf.GPUOptions(
    allow_growth=True # True->必要になったら確保, False->全部
  )
)

with detection_graph.as_default():
  with tf.Session(graph=detection_graph, config=config) as sess:
    index = 0
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    #for image_path in TEST_IMAGE_PATHS:
    for dirpath, dirnames, filenames in os.walk(PATH_TO_TEST_IMAGES_DIR):
      for filename in filenames:
        #try:    
          (fn,ext) = os.path.splitext(filename)
          if ext.upper() not in exts:        
            continue
          image_path = os.path.join(dirpath, filename)
          print(image_path)
          # https://stackoverflow.com/questions/45400346/valueerror-cannot-reshape-array-of-size-357604-into-shape-299-299-3
          image = Image.open(image_path).convert('RGB')
          # the array based representation of the image will be used later in order to prepare the
          # result image with boxes and labels on it.
          image_np = None
          image_np = load_image_into_numpy_array(image)
          image_np_cp = copy.deepcopy(image_np)
          # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
          image_np_expanded = np.expand_dims(image_np, axis=0)
          # Actual detection.
          (boxes, scores, classes, num) = sess.run(
              [detection_boxes, detection_scores, detection_classes, num_detections],
              feed_dict={image_tensor: image_np_expanded})
          # Visualization of the results of a detection.
          items, image_pil, box_to_display_str_map = vis_util.visualize_boxes_and_labels_on_image_array(
              image_np,
              np.squeeze(boxes),
              np.squeeze(classes).astype(np.int32),
              np.squeeze(scores),
              category_index,
              use_normalized_coordinates=True,
              line_thickness=8)
          # Bounding box images are saved
          output_image_path = os.path.join(PATH_TO_OUTPUT_IMAGES_DIR, filename)
          print(output_image_path)
          plt.figure(figsize=IMAGE_SIZE, dpi=300) # dpiいじったら文字が読めるようになる
          plt.imshow(image_np)
          #plt.savefig(output_image_path) # ここを追加
          i = 0
          im_width, im_height = image_pil.size
          # Bounding box images are cropped
          for box, color in items:
            box_to_display_str = next(filter(None, box_to_display_str_map[box]), None)
            print(box_to_display_str)
            if box_to_display_str is None:
              continue
            target_key, accuracy_str = box_to_display_str.split(':')
            accuracy = int(re.sub('\s|%', '', accuracy_str))
            print(target_key)
            print(FLAGS.target_dir)
            if target_key == FLAGS.target_dir and accuracy >= pass_accuracy_rate:
              ymin, xmin, ymax, xmax = box
              (xminn, xmaxx, yminn, ymaxx) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
              cropped_image = tf.image.crop_to_bounding_box(image_np_cp, int(yminn), int(xminn), 
                                           int(ymaxx - yminn), int(xmaxx - xminn))
              cropped_image_encoded = None
              if ext == 'png':
                cropped_image_encoded = tf.image.encode_png(cropped_image)
              else:
                cropped_image_encoded = tf.image.encode_jpeg(cropped_image) 
              crop_image_path = os.path.join(PATH_TO_CROP_IMAGES_DIR, '{}_{}{}'.format(fn, i, ext))
              print(accuracy_str)
              print(crop_image_path)
              #file = tf.write_file(tf.constant(crop_image_path), cropped_image_encoded)
              result = sess.run(tf.write_file(tf.constant(crop_image_path), cropped_image_encoded))
              i+=1
              index += 1
            #else:
            #  print("######## anonter image")
            #  print(target_key)
        #except:
        #  print("!!! Exception !!!! {}:{}".format(filename, sys.exc_info()))
    print('cropped {}: {} files'.format(FLAGS.target_dir, index))
# In[   ]:
