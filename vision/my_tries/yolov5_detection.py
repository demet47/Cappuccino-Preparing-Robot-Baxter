import torch
import cv2
from matplotlib import pyplot as plt

# Load the model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Load the image
image = cv2.imread('image_0.png')

# Detect objects
results = model(image)

# Visualize the results
results.show()

# Get the bounding box coordinates
boxes = results.xyxy[0].tolist()

# Print the bounding box coordinates
for box in boxes:
    print(box)

