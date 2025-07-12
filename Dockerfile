# Use CUDA-enabled PyTorch base image
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create a directory for model checkpoints
RUN mkdir -p /app/checkpoints

# Set the entrypoint to run the training script
ENTRYPOINT ["python", "train.py"]

# Note: When running the container, mount a volume to /app/checkpoints to persist model checkpoints
# Example: docker run --gpus all -v $(pwd)/checkpoints:/app/checkpoints spark-sched-sim
