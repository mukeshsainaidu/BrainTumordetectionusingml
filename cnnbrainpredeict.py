

import pandas as pd
import numpy as np
import tensorflow
from tensorflow import keras
import matplotlib.pyplot as plt
from keras.preprocessing.image import  ImageDataGenerator as Imgen
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, Activation
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
cnn = load_model('my_model.h5')
from matplotlib.pyplot import imshow
from PIL import Image, ImageOps
data = np.ndarray(shape=(1, 130, 130, 3), dtype=np.float32)
image = Image.open("image(5).jpg")
size = (130, 130)
image = ImageOps.fit(image, size, Image.ANTIALIAS)
image_array = np.asarray(image)
#cv2.imshow('image1',image)
normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
data[0] = normalized_image_array
prediction = cnn.predict(data)
print(prediction)
predict_index = np.argmax(prediction)
print(predict_index)
if predict_index==0:
    print('glioma_tumor')
elif predict_index==1:
    print('meningioma_tumor')
elif predict_index==2:
    print('no_tumor')
elif predict_index==3:
    print('pituitary_tumor')
else:
    print('unknow class')
    

