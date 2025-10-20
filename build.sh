# Build the image
docker build -t yolov8-training-custom .

# Run container
docker run -it \
  --name bdd100k_assignment \
  -v $(pwd):/workspace \
  -p 8888:8888 \
  yolov8-training-custom \
  /bin/bash