
FROM python:3.10-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables to limit memory usage
ENV OMP_NUM_THREADS=2
ENV MKL_NUM_THREADS=2
ENV TORCH_NUM_THREADS=2

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files and models
COPY app.py .
COPY best_resnet18_sdnet.pth .
COPY best_unet_v2.pth .

# Expose port
EXPOSE 7860

# Run the app
CMD ["python", "app.py"]
