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
        "transfer_speed": payload_size / transfer_time
    }
}

append_to_csv = {
    "executed_at": start_time,
    "total_images": len(payload),
    "total_payload_size": payload_size,
    "transfer_speed": result['payload']['transfer_speed'],
    "executed_on": url,
    "execution_time_on_service": result['payload']['total_execution_time_on_server'],
    "total_time": end_time - start_time + result['payload']['total_execution_time_on_server'],
}

with open('executions_history.csv', mode="a") as csv_file:
    a = append_to_csv
    write_string = str(time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(a['executed_at'])))
    write_string += "," + str(a['total_images'])
    write_string += "," + str(a['total_payload_size'] / 1000) + "MB"
    write_string += "," + str(a['transfer_speed'] / 1000) + "MB/s"
    write_string += "," + str(a['executed_on'])
    write_string += "," + str(a['execution_time_on_service'])
    write_string += "," + str(a['total_time'])
    write_string += "\n"
    csv_file.write(str(write_string))

# Process the result as needed
print(result)