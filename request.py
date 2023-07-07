import sys

import requests
import tensorflow as tf
import os
import time

folder_path = 'object-detection-SMALL_2_files'
base64_strings = []
names = []
for filename in os.listdir(folder_path):
    # Read all the files in a folder
    temp_path = os.path.join(folder_path, filename)
    # Read file with tensorflow
    image_data = tf.io.read_file(temp_path)
    # Encode the tensor as base64
    encoded_tensor = tf.io.encode_base64(image_data)
    # Decode the byte string to UTF-8 string
    base64_string = encoded_tensor.numpy().decode('utf-8')

    base64_strings.append(base64_string)

# Create the request payload
payload = base64_strings
# Send the POST request to the API
# url = 'http://localhost'
url = 'http://ec2-44-212-28-11.compute-1.amazonaws.com'

start_time = time.time()
response = requests.post(url + ':5000/api/detect', json=payload)
# Get the JSON response
response = response.json()
end_time = time.time()
payload_size = 0
for img in base64_strings:
    payload_size = payload_size + sys.getsizeof(img)

transfer_time = response['payload_received_at'] - start_time

result = {
    "image_objects": response['image_objects'],
    "payload": {
        "total_execution_time_on_server": response['total_execution_time'],
        "transfer_speed": str(payload_size / transfer_time) + " B/s"
    }
}
# Process the result as needed
print(result)