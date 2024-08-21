# Use the smaller slim image as a base
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Install system dependencies and Python packages
COPY requirements.txt /app/
RUN apt-get update && apt-get install -y --no-install-recommends \
    libatlas-base-dev \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy the application code and necessary files into the container
COPY . /app/

# Expose port 8000 for the Flask application
EXPOSE 8000

# Set the command to run the Flask application
CMD ["flask", "run", "--host=0.0.0.0", "--port=8000"]