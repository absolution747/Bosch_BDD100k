# Use Ultralytics official CPU image as base
FROM ultralytics/ultralytics:latest-cpu

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /workspace

# Update system packages
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt /workspace/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /workspace/requirements.txt

# Copy project files
COPY . /workspace/

# Make download script executable
RUN if [ -f /workspace/download_data.sh ]; then \
    chmod +x /workspace/download_data.sh; \
    fi

# Expose Jupyter notebook port
EXPOSE 8888

# Set default command
CMD ["/bin/bash"]