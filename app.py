import base64

from flask import Flask, request, jsonify
import json
import tensorflow as tf
import numpy as np
from flask import Flask, request, Response, jsonify
import tensorflow as tf
import tensorflow_hub as hub
# For drawing onto the image.
import numpy as np
import json

app = Flask(__name__)


def detection_loop(detector, images):
    print(f"Loop entered, yes")
    # print(f"Images is {images}")
    res = []
    # Access the 'images' list from the data dictionary
    try:
        for image in images:
            # Replace symbols to make it Web-safe for tensorflow
            img = image.replace('/', '_').replace('+', '-')

            # Decode the byte string to a TensorFlow tensor
            decoded_img = tf.io.decode_base64(img)
            # print(f"Decoded image is {decoded_img}")
            decoded_img = tf.image.decode_jpeg(decoded_img, channels=3)
            decoded_img = tf.image.convert_image_dtype(decoded_img, tf.float32)[tf.newaxis, ...]
            result = detector(decoded_img)
            classes = np.unique(result["detection_class_entities"]).astype(str).tolist()
            res.append(classes)

    except Exception as e:
        return {"error": str(e)}

    # Convert the list to JSON string
    json_data = json.dumps(res)
    # Return JSON data
    return json_data


# Routing HTTP posts to this method
@app.route('/api/detect', methods=['POST', 'GET'])
def main():
    images = request.get_json(force=True)
    # Perform object detection on images
    module_handle = "https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1"
    detector = hub.load(module_handle).signatures['default']

    result = detection_loop(detector, images)
    return result


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
