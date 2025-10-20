# Build the image
docker build -t yolov8-training-custom .

# Run container
docker run -it \
  -v $(pwd):/workspace \
  -p 8888:8888 \
  yolov8-training-custom

# Run with Jupyter notebook
docker run -it \
  -v $(pwd):/workspace \
  -p 8888:8888 \
  yolov8-training-custom \
  jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root