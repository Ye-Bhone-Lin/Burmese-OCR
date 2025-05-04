import numpy as np
import cv2
import matplotlib.pyplot as plt
import os 
from keras.models import load_model

model = load_model('D:\Burmese-OCR\src\model\CNN.keras')

test_image_path = os.getcwd() + r"/images.jpg"

img = cv2.imread(test_image_path)
img = cv2.resize(img, (224, 224))

img_array = np.array(img, dtype=np.float32)
img_array = np.expand_dims(img_array, axis=0)

pred = model.predict(img_array)
print("Prediction:", pred)
