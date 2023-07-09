import sys

import requests
import tensorflow as tf
import os
import time

folder_path = 'object-detection-SMALL'
base64_strings = []
names = []
for filename in os.listdir(folder_path)[:5]:
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
url = 'http://localhost'
# url = 'http://ec2-54-174-141-192.compute-1.amazonaws.com'

start_time = time.time()
response = requests.post(url + ':5000/api/detect', json=payload)
# Get the JSON response
response = response.json()
stats = response['stats']
append_to_csv = {
    "executed_at": str(stats['executed_at']),
    "total_images": str(stats['total_images']),
    "total_payload_size": stats['total_payload_size'],
    "transfer_speed": stats['transfer_speed'],
    "execution_time_on_service": str(round(stats['total_execution_time'], 2)),
    "avg_inference_time": str(stats["avg_inference_time"]),
    "url": url
}
#
with open('executions_history.csv', mode="a") as csv_file:
    a = append_to_csv
    append_string = ""
    for key in append_to_csv:
        append_string += append_to_csv[key] + ","

    append_string = append_string[:-1] + "\n"
    csv_file.write(str(append_string))

# Process the result as needed
print(response)