import keras
from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, InputLayer, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D
from keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import sys
import gc

train = pd.read_csv('train.csv')

new_dat = pd.DataFrame()
train_image = []
train_tags = []

gc.collect()
for i in tqdm(range(15001, 20001)):
    img = image.load_img('training_images/' + train['image'][i])
    train_image.append(image.img_to_array(img) / 255)
    train_tags.append(train['class'][i])
    del img
    if i % 500 == 0:
        gc.collect()

new_dat["image"] = train_image
new_dat["class"] = train_tags
new_dat.to_csv("/train_images_data3.csv", header=True, index=False)
