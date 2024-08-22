# Image Similarity Web Service

## Overview
This project is a web service that accepts an image URL as input and returns up to 10 images that are most similar to the input image. The web service is built using Flask and utilizes a pre-trained deep learning model (ResNet50) for feature extraction and FAISS (Facebook AI Similarity Search) for efficient similarity search.

The service is containerized using Docker, and the image is optimized to be as small as possible while ensuring all dependencies are met. Additionally, the logic for building the FAISS index and feature extraction is available in a [Google Colab notebook](https://colab.research.google.com/drive/1bi-Tjnd6ph_31-Tu6u8buMVqMuKQ97Ga?usp=drive_link).

## Features
**Image Similarity Search:** Takes an image URL as input and returns a list of similar images.

**Pre-trained Model:** Uses [ResNet50](https://pytorch.org/hub/nvidia_deeplearningexamples_resnet50/), a state-of-the-art deep learning model for image feature extraction.

**Efficient Search with FAISS:** Utilizes FAISS for fast and scalable similarity search.

**Containerized with Docker:** The web service runs in a Docker container, making it easy to deploy and maintain.

## Dependencies
- Python 3.10
- Docker
- Flask
- FAISS
- Torch (PyTorch)

## Project Structure

- `app.py`                 # Main Flask application
- `dockerfile`             # Docker configuration file
- `requirements.txt`       # Python dependencies
- `data/`                  # Directory for static data files
  - `faiss_index.index`    # Precomputed FAISS index file
  - `image_urls.pkl`       # Pickle file containing image URLs
- `.dockerignore`          # Files and directories to ignore in Docker build context
- `.gitignore`             # Ignore virtual environment and other unnecessary files
- `README.md`              # Project documentation

## Installation and Setup

1. Clone the Repository

````
git clone https://github.com/dintellect/SimilarImageSearch.git
cd SimilarImageSearch
````

2. Prepare the Environment
Ensure you have Python 3.10 installed on your system.

- Run the following command to create a virtual environment.

````
python3.10 -m venv sim_env
````

- Activate Environment

````
source venv/bin/activate
````

- Install Dependencies in Env
````
pip install -r requirements.txt
````

- Deactivate the Virtual Environment
````
deactivate
````

3. Install Dependencies
Install the required Python packages using `pip`:
````
pip install -r requirements.txt
````

4. Run the Application
You can start the Flask application by running:

````
python app.py
````

5. Build and Run with Docker
To build the Docker image:

````
docker build -t image-similarity-service .
````

Run the Docker container:

````
docker run -d -p 8000:8000 image-similarity-service
````

## Usage

### API Endpoint

- **URL:** /find_similar_images

- **Method:** POST

- **Request Body:** JSON object with a single key image_url.

Example:

````
{
  "image_url": "https://example.com/sample_image.jpg"
}
````
- **Response:** JSON object with a list of URLs of similar images.

Example:

````
{
  "similar_images": [
      "https://example.com/similar_image1.jpg",
      "https://example.com/similar_image2.jpg",
      ...
  ]
}
````

## Testing with cURL 
```
curl -X POST "http://127.0.0.1:8000/find_similar_images" -H "Content-Type: application/json" -d '{"image_url": "http://example.com/image.jpg"}'
```

## Testing the API with Postman
- Open Postman and create a new POST request.
- Set the request URL to `http://localhost:8000/find_similar_images`.
- In the body tab, choose `raw` and `JSON`.
- Provide the JSON input as described above.
- Click "Send" to receive the list of similar images.

## FAISS Index Creation and Feature Extraction (Google Colab Notebook)

For understanding the feature extraction and FAISS index creation, you can refer to the Google Colab notebook:

[FAISS Index Creation and Feature Extraction - Google Colab](https://colab.research.google.com/drive/1bi-Tjnd6ph_31-Tu6u8buMVqMuKQ97Ga?usp=drive_link)

This notebook covers:

- Downloading and preprocessing the dataset.
- Using the pre-trained ResNet50 model for feature extraction.
- Creating and saving a FAISS index for fast similarity search.
- Pickling the URLS list

## Dockerfile Explanation
The Dockerfile is optimized for a minimal footprint by using `python:3.10-slim` as the base image. Key dependencies like FAISS, PyTorch, and Flask are installed, and unnecessary files are excluded using `.dockerignore.`

### Key Sections:
**Base Image:** python:3.10-slim is used to keep the image lightweight.

**System Dependencies:** Required system packages are installed using apt-get.

**Python Dependencies:** Installed via pip from requirements.txt.

**Files Copy:** The FAISS index and image URLs pickle file are copied into the container.

**Expose Port:** The Flask application runs on port 8000.

## Dependencies Explaination

#### faiss-cpu==1.7.3

**Description:** A library for efficient similarity search and clustering of dense vectors.
**Purpose:** Used for building and querying the similarity index.
#### python-multipart==0.0.5

**Description:** Supports multipart encoding for handling file uploads.
**Purpose:** Enables the Flask application to handle file uploads via POST requests.

#### Direct Wheel Files for PyTorch and torchvision:

https://download.pytorch.org/whl/cpu/torch-2.0.0%2Bcpu-cp310-cp310-linux_x86_64.whl: PyTorch optimized for CPU.

https://download.pytorch.org/whl/cpu/torchvision-0.15.0%2Bcpu-cp310-cp310-linux_x86_64.whl: Torchvision optimized for CPU.

**Purpose:** Ensures that only the CPU-optimized versions of PyTorch and torchvision are installed, avoiding unnecessary GPU-related components. This approach helps to reduce the size of the Docker image and speeds up the installation process by using precompiled wheel files.


## References

- [Faiss: A library for efficient similarity search](https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/)

- [Exploring ResNet50](https://medium.com/@nitishkundu1993/exploring-resnet50-an-in-depth-look-at-the-model-architecture-and-code-implementation-d8d8fa67e46f)

- [MobileNet V2](https://huggingface.co/docs/transformers/en/model_doc/mobilenet_v2)
