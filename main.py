import cv2     # for capturing videos
import math   # for mathematical operations
import matplotlib.pyplot as plt    # for plotting the images
import pandas as pd
from keras.preprocessing import image   # for preprocessing the images
import numpy as np    # for mathematical operations
from keras.utils import np_utils
from skimage.transform import resize   # for resizing images
from sklearn.model_selection import train_test_split
from glob import glob
from tqdm import tqdm
import os
import random
import itertools
import shutil

files = os.listdir("data/")
training = pd.DataFrame()
test = pd.DataFrame()
training_list = []
test_list = []
training_tags = []
test_tags = []

for f in files:
    temp = pd.DataFrame()
    temp_l = []
    temp_l_c = []
    dir = os.listdir("data/" + f)
    for dir_f in dir:
        temp_l.append(dir_f)
        temp_l_c.append(f)
    X = temp_l
    y = temp_l_c
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    for x in X_train:
        training_list.append(x)
    for x in X_test:
        test_list.append(x)
    for y in y_train:
        training_tags.append(y)
    for y in y_test:
        test_tags.append(y)

training["name"] = training_list
training["class"] = training_tags
test["name"] = test_list
test["class"] = test_tags

for i in range(training.shape[0]):
    if not os.path.exists("training/" + training["class"][i]):
        os.makedirs(training["class"][i])
    if not os.path.exists("test/" + test["class"][i]):
        os.makedirs(test["class"][i])

# path = 'data_1/'
# for i in tqdm(range(training.shape[0])):
#     x = 0
#     video = training["name"][i]
#     cl = training["class"][i]
#     cap = cv2.VideoCapture("data/" + cl + "/" + video)
#
#     ret, frame = cap.read()
#     while ret:
#
#         filename = video.split(".")[0] + "_frame_" + str(x) + '.jpg'
#         print("capturing frame " + str(x) + " from video " + video)
#         fullname = os.path.join(path, filename)
#         cv2.imwrite(fullname, frame)
#         ret, frame = cap.read()
#         x += 1