# Use Python base image
FROM python:3.10

# Prevent large OpenCV logs
ENV OPENCV_VIDEOIO_DEBUG=0

# Install system dependencies (required for cv2, dlib, mediapipe)
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean

# Create app directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install pip deps (including mediapipe!)
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the whole project
COPY . .

# Railway uses PORT environment variable
ENV PORT=8080

# Start the app
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]
