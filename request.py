import requests
import tensorflow as tf
import os


folder_path='object-detection-SMALL_2_files'
base64_strings=[]
names=[]
for filename in os.listdir(folder_path):
    #Read all the files in a folder
    temp_path = os.path.join(folder_path, filename)
    # Read file with tensorflow
    image_data=tf.io.read_file(temp_path)
    # Encode the tensor as base64
    encoded_tensor = tf.io.encode_base64(image_data)
    # Decode the byte string to UTF-8 string
    base64_string = encoded_tensor.numpy().decode('utf-8')
    
    base64_strings.append(base64_string)

    #Add filename
    names.append(filename)



# Create the request payload
payload = {
    'images':base64_strings,
    'names' :names
}

print("About to send request")
# Send the POST request to the API
response = requests.post('http://localhost:5000/api/detect', json=payload)

# Get the JSON response
result = response.json()

# Process the result as needed
print(result)