# Dockerfile for the Spleen Segmentation API
FROM python:3.11-slim

# Installation of system dependencies (necessary for numpy/onnx)
RUN apt-get update && apt-get install -y \
	libgomp1 \
	&& rm -rf /var/lib/apt/lists/*

# Working directory
WORKDIR /app

# Copy of requirements and installation
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy of code source and model
COPY src/ src/
COPY outputs/model_spleen.onnx outputs/model_spleen.onnx

# Exposition of port
EXPOSE 8000

# Command to launch
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]