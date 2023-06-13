import requests
import json
import base64
import tensorflow as tf
import io


# Encode the image data as base64
image_data=tf.io.read_file('object-detection-SMALL_2_files\\000000150805.jpg')
# Encode the tensor as base64
encoded_tensor = tf.io.encode_base64(image_data)

# Decode the byte string to UTF-8 string
base64_string = encoded_tensor.numpy().decode('utf-8')



#print(f"Decoded string is {base64_string}")

# Create the request payload
payload = {
    'images': [base64_string]
}

print("About to send request")
# Send the POST request to the API
response = requests.post('http://localhost:5000/api/detect', json=payload)

# Get the JSON response
result = response.json()

# Process the result as needed
print(result)