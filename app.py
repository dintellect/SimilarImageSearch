from flask import Flask, request, jsonify
from typing import List
import faiss
import numpy as np
from PIL import Image
import torch
from torchvision import models, transforms
import pickle
from PIL import UnidentifiedImageError
import io
import requests

# Initialize the Flask app
app = Flask(__name__)

# Load the pre-trained ResNet50 model with weights and set it to evaluation mode
model = models.resnet50(weights=True)
model.eval()

# Define the image transformation pipeline to preprocess images for the model
transform = transforms.Compose([
    transforms.Resize(256),  # Resize the image to 256x256 pixels
    transforms.CenterCrop(224),  # Crop the center 224x224 pixels
    transforms.ToTensor(),  # Convert the image to a tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize the tensor
])

# Load the Faiss index and image URLs from files
index = faiss.read_index("data/faiss_index.index")  # Load the pre-built Faiss index for fast similarity search
with open('data/image_urls.pkl', 'rb') as f:
    image_urls = pickle.load(f)  # Load the list of image URLs corresponding to the index

def load_image(url: str) -> Image.Image:
    """Fetch and load an image from a URL."""
    try:
        response = requests.get(url, timeout=10)  # Fetch the image from the URL with a 10-second timeout
        img = Image.open(io.BytesIO(response.content))  # Open the image from the response content
        return img
    except (requests.exceptions.RequestException, UnidentifiedImageError, ValueError) as e:
        print(f"Error loading image from {url}: {e}")  # Log the error
        return None  # Return None if there's an error loading the image

def extract_features(image: Image.Image) -> np.ndarray:
    """Extract feature vector from the image using ResNet50."""
    image = transform(image).unsqueeze(0)  # Apply transformations and add batch dimension
    with torch.no_grad():  # Disable gradient calculation for inference
        features = model(image)  # Get features from the model
    return features.numpy().flatten()  # Flatten the features into a 1D array

def normalize(x: np.ndarray) -> np.ndarray:
    """Normalize the feature vector to unit length."""
    return x / np.linalg.norm(x, ord=2, axis=1, keepdims=True)  # Normalize the feature vector

@app.route('/find_similar_images/', methods=['POST'])
def find_similar_images():
    """Endpoint to find and return similar images based on input image URL."""
    data = request.get_json()  # Get JSON data from the request
    image_url = data.get('image_url')  # Extract the image URL from the JSON data

    if not image_url:
        return jsonify({"error": "No image_url provided"}), 400  # Return an error if no URL is provided

    input_image = load_image(image_url)  # Load the image from the provided URL
    if input_image is None:
        return jsonify({"error": "Invalid image URL"}), 400  # Return an error if the image could not be loaded

    input_features = extract_features(input_image)  # Extract features from the input image
    input_features = normalize(input_features.reshape(1, -1))  # Normalize the feature vector

    # Perform the search to find the top 11 most similar images
    _, indices = index.search(input_features, 11)

    # Retrieve the URLs of the top similar images
    similar_images = [image_urls[i] for i in indices[0]][1:]
    return jsonify({"similar_images": similar_images})  # Return the list of similar image URLs as JSON

# Entry point for running the Flask app
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)  # Run the app on all available IP addresses, port 8000
