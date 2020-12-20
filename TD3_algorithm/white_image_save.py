import scipy.misc
import numpy as np
from matplotlib.image import imread
import matplotlib
import cv2
from PIL import Image

image_paths = ["images/mask.png", "images/MASK1.png", "images/sand.jpg"]
for image_path in image_paths:
    img = cv2.imread(image_path)    
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grayImage.fill(255.0)
    cv2.imwrite(image_path, grayImage)
