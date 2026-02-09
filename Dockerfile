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

# Copy source and install package
COPY src/ src/
COPY api/ api/
COPY pyproject.toml .
RUN pip install -e .

# Copy model (must exist; run export_onnx first or mount at runtime)
COPY outputs/ outputs/

EXPOSE 8000

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]