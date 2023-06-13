from flask import Flask, request, jsonify
import json
import os
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
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

app = Flask(__name__)

#def load_img(path):
#    img = tf.io.read_file(path)
#    img = tf.image.decode_jpeg(img, channels=3)
#    return img

def detection_loop(detector, images):
    print(f"Loop entered, yes")
    #print(f"Images is {images}")
    r = []
    # Access the 'images' list from the data dictionary
    images_list = images['images']
    for img in images_list:
        #print(f"String before decoding is {img}")
        #for filename in os.listdir(folder_path):
        #img_path = os.path.join(folder_path, filename)
        #img = load_img(img)
        #img = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
        #result = detector(converted_img)
        # Decode the base64 image
        # Check if padding is needed
        #padding_needed = len(img) % 4 != 0
        #if padding_needed:
        #    padding_length = 4 - (len(img) % 4)
        #    img += '=' * padding_length
        # Decode the base64 image
        # Assume encoded_image is the encoded image string
        # Convert the base64 string to a byte string
        byte_string = img.encode('utf-8')

        # Decode the byte string to a TensorFlow tensor
        decoded_img = tf.io.decode_base64(byte_string)
        #print(f"Decoded image is {decoded_img}")
        decoded_img = tf.image.decode_jpeg(decoded_img, channels=3)
        decoded_img = tf.image.convert_image_dtype(decoded_img, tf.float32)[tf.newaxis, ...]
        #print(f"img is {img}")
        #img = tf.io.read_file(img_data)
        #img = tf.image.decode_jpeg(img, channels=3)
        #converted_img  = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
        #img = Image.open(io.BytesIO(img_data))
        
        # Convert image to tensor
        #img = tf.convert_to_tensor(np.array(img))
        #img = tf.expand_dims(img, axis=0)
        #img = tf.image.convert_image_dtype(img, tf.float32)
        
        result = detector(decoded_img)
        classes = np.unique(result["detection_class_entities"]).astype(str).tolist()
        r.append(classes)

    # Convert the list to JSON string
    json_data = json.dumps(r)
    print(json_data)
    # Return JSON data
    return json_data

# Routing HTTP posts to this method
@app.route('/api/detect', methods=['POST', 'GET'])
def main():
    images = request.get_json(force=True)
    print("Request received")
    print(images)
    # Get the array of images from the JSON body
    #imgs = data['images']

    # Prepare images for object detection
    #images = []
    #for img in imgs:
    #    images.append(np.array(Image.open(io.BytesIO(base64.b64decode(img))), dtype=np.float32))

    # Perform object detection on images
    module_handle = "https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1" 
    detector = hub.load(module_handle).signatures['default']
    #folder_path = 'object-detection-SMALL_2_files'

    result = detection_loop(detector, images)
    return result

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
