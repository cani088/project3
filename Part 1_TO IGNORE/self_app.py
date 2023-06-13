
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from PIL import Image
from PIL import ImageDraw
import os
import detect
import tflite
import platform
import datetime
import cv2
import time
import numpy as np
import io
from io import BytesIO
from flask import Flask, request, Response, jsonify
import random
import re
import tensorflow as tf
import tensorflow_hub as hub
# For downloading the image.
import matplotlib.pyplot as plt
import tempfile
from six.moves.urllib.request import urlopen
from six import BytesIO
# For drawing onto the image.
import numpy as np
from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps
import json
# For measuring the inference time.
import time

module_handle = "https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1" 
detector = hub.load(module_handle).signatures['default']

# def download_and_resize_image(url, new_width=256, new_height=256, display=False):
#     # Check if the image has already been downloaded
#     _, filename = tempfile.mkstemp(suffix=".jpg")
#     response = urlopen(url)
#     image_data = response.read()
#     image_data = BytesIO(image_data)
#     pil_image = Image.open(image_data)
#     pil_image = ImageOps.fit(pil_image, (new_width, new_height), Image.ANTIALIAS)
#     pil_image_rgb = pil_image.convert("RGB")
#     pil_image_rgb.save(filename, format="JPEG", quality=90)
#     print("Image downloaded to %s." % filename)
#     return filename


def load_img(path):
  img = tf.io.read_file(path)
  img = tf.image.decode_jpeg(img, channels=3)
  return img


def detection_loop(detector, folder_path):
    r=[]
    for filename in os.listdir(folder_path):
        #img = download_and_resize_image(url, 1280, 856, True)
        #result = {key:value.numpy() for key,value in result.items()}
        #print("Found %d objects." % len(result["detection_scores"]))
        #print(result["detection_class_entities"])
        # Print available attributes in the result dictionary
        #r.append(result["detection_class_entities"])
        img = os.path.join(folder_path, filename)
        img = tf.io.read_file(img)
        img = tf.image.decode_jpeg(img, channels=3)
        converted_img  = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
        start_time = time.time()
        result = detector(converted_img)
        end_time = time.time()
        r.append(np.unique(result["detection_class_entities"]).astype(str).tolist())
        print(np.unique(result["detection_class_entities"]).astype(str).tolist())

    # Convert the list to JSON string
    json_data = json.dumps(r)
    # Return JSON data with appropriate headers
    return Response(json_data, mimetype='application/json')
folder_path='object-detection-SMALL_2_files'
# current_path = os.getcwd()
# print("Current working directory:", current_path)
#images_url = ["https://upload.wikimedia.org/wikipedia/commons/6/60/Naxos_Taverna.jpg", "https://upload.wikimedia.org/wikipedia/commons/0/09/The_smaller_British_birds_%288053836633%29.jpg"]  #@param
#folder_path='C:\\Users\\awtfh\\Documents\\GitHub\\project3\\object-detection-SMALL_2_files'

r1= detection_loop(detector, folder_path)
print(r1)