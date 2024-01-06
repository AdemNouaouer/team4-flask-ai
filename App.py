from flask import Flask, render_template, request, jsonify
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cloudinary
from cloudinary import uploader, config
import requests
from io import BytesIO

app = Flask(__name__)

# Configure Cloudinary with your API credentials
cloudinary.config(
    cloud_name='dk4dqgocn',
    api_key='634939829644144',
    api_secret='FvBFVaePzLAXk7WHA09fbJWXa5w'
)

# Load the pre-trained model
model = load_model('model.h5')
class_labels = ['Class_0', 'Class_1']

# Function to preprocess the input image
def preprocess_image(img_path):
    # Load the image using PIL
    img = Image.open(img_path)
    
    # Resize the image to the target size (64, 64)
    img = img.resize((64, 64))
    
    # Convert the image to a NumPy array
    img_array = np.array(img)
    
    # Expand the dimensions to create a batch of size 1
    img_array = np.expand_dims(img_array, axis=0)
    
    # Normalize pixel values to be between 0 and 1
    img_array = img_array / 255.0
    
    return img_array

# Function to make predictions
def predict_image_class(img_path):
    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class])  # Convert to Python float
    class_label = class_labels[predicted_class]

    return class_label, confidence

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        # Upload the file to Cloudinary
        cloudinary_response = uploader.upload(file)

        # Access the Cloudinary URL of the uploaded file
        cloudinary_url = cloudinary_response['url']

        # Download the image to a temporary file
        response = requests.get(cloudinary_url)
        img = Image.open(BytesIO(response.content))
        temp_file_path = 'temp_image.jpg'
        img.save(temp_file_path)

        # Make predictions
        predicted_class, confidence = predict_image_class(temp_file_path)

        # Remove the temporary file
        os.remove(temp_file_path)

        return jsonify({
            'image_path': cloudinary_url,
            'predicted_class': predicted_class,
            'confidence': confidence
        })

    except Exception as e:
        # Handle prediction error
        error_message = f'Error during prediction: {str(e)}'
        return jsonify({'error': error_message})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
