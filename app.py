from flask import Flask, request
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import json
import time

app = Flask(__name__)


def detection_loop(detector, images):
    print(f"Loop entered, yes")
    res = []
    try:
        for image in images:
            # Replace symbols to make it Web-safe for tensorflow
            img = image.replace('/', '_').replace('+', '-')

            # Decode the byte string to a TensorFlow tensor
            decoded_img = tf.io.decode_base64(img)
            # Decode the JPEG image and convert it to float32 format
            decoded_img = tf.image.decode_jpeg(decoded_img, channels=3)
            decoded_img = tf.image.convert_image_dtype(decoded_img, tf.float32)[tf.newaxis, ...]

            # Perform detection using the provided detector
            result = detector(decoded_img)

            # Extract the detected classes and convert them to a list of strings
            classes = np.unique(result["detection_class_entities"]).astype(str).tolist()

            # Append the detected classes to the result list
            res.append(classes)

    except Exception as e:
        return {"error": str(e)}

    # Convert the list to JSON string
    json_data = json.dumps(res)
    # Return JSON data
    return json_data


# Routing HTTP posts to this method
@app.route('/api/detect', methods=['POST'])
def main():
    start_time = time.time()
    images = request.get_json(force=True)
    # Perform object detection on images
    module_handle = "https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1"
    detector = hub.load(module_handle).signatures['default']

    result = detection_loop(detector, images)
    end_time = time.time()

    return {"image_objects": result, "inference_time": end_time - start_time}


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
