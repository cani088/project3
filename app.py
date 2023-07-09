import sys

from flask import Flask, request
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import json
import time

app = Flask(__name__)


def detection_loop(detector, images):
    res = []
    execution_times = []
    try:
        for image in images:
            # Record the time when the image started processing
            start_time = time.time()
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
            # Record the time when the image finished processing
            end_time = time.time()
            # Add the recorded time to the list of times
            execution_times.append(end_time - start_time)

    except Exception as e:
        return {"error": str(e)}

    # Convert the list to JSON string
    json_data = json.dumps(res)
    # Return JSON data which consists the objects found in the images and the average inference time for image processing
    return json_data, sum(execution_times) / len(execution_times)


# Routing HTTP posts to this method
@app.route('/api/detect', methods=['POST'])
def main():
    # Record the time when the request was received
    received_at = time.time()
    images = request.get_json(force=True)
    # Get the total size of the request
    payload_size = int(request.headers.get('Content-Length'))
    # Calculate how long it took from when we received the request until the request was fully transferred and processed
    transfer_time = time.time() - received_at
    # Calculate the speed
    transfer_speed = (payload_size / 1000000) / transfer_time
    transfer_speed = str(round(transfer_speed, 2)) + "MB/s"
    # Perform object detection on images
    module_handle = "https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1"
    detector = hub.load(module_handle).signatures['default']

    objects, avg_inference_time = detection_loop(detector, images)

    # Return response to user
    return {
        "stats": {
            "executed_at": str(time.strftime('%d-%m-%Y %H:%M:%S', time.gmtime(received_at))),
            "total_images": len(images),
            "total_payload_size": str(round(payload_size / 1000000, 2)) + "MB",
            "transfer_speed": transfer_speed,
            "avg_inference_time": round(avg_inference_time, 2),
            "total_execution_time": time.time() - received_at
        },
        "image_objects": objects,
    }


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
