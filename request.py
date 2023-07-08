import sys

import requests
import tensorflow as tf
import os
import time

folder_path = 'object-detection-SMALL'
base64_strings = []
names = []
for filename in os.listdir(folder_path)[:2]:
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
end_time = time.time()
payload_size = 0
for img in base64_strings:
    payload_size = payload_size + sys.getsizeof(img)

transfer_time = response['payload_received_at'] - start_time

result = {
    "image_objects": response['image_objects'],
    "payload": {
        "total_execution_time_on_server": response['total_execution_time'],
        "transfer_speed": (payload_size / 1000000) / transfer_time
    }
}

append_to_csv = {
    "executed_at": str(time.strftime('%d-%m-%Y %H:%M:%S', time.gmtime(start_time))),
    "total_images": str(len(payload)),
    "total_payload_size": str(round(payload_size / 1000000, 2)) + "MB",
    "transfer_speed": str(round(result['payload']['transfer_speed'], 2)) + "MB/s",
    "executed_on": url,
    "execution_time_on_service": str(round(result['payload']['total_execution_time_on_server'], 2)) + " seconds",
    "total_time": str(round(end_time - start_time + result['payload']['total_execution_time_on_server'], 2)) + " seconds",
    "avg_inference_time": str(round(response["avg_inference_time"], 2)) + " seconds"
}

with open('executions_history.csv', mode="a") as csv_file:
    a = append_to_csv
    append_string = ""
    for key in append_to_csv:
        append_string += append_to_csv[key] + ","

    append_string = append_string[:-1] + "\n"
    csv_file.write(str(append_string))

# Process the result as needed
print(result)