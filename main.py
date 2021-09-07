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
from pathlib import Path

if len(os.listdir('training/Acorn_Woodpecker')) == 0:
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
            Path('training/' + training["class"][i]).mkdir(parents=True, exist_ok=True)
        if not os.path.exists("test/" + training["class"][i]):
            Path('test/' + training["class"][i]).mkdir(parents=True, exist_ok=True)
        shutil.copy("data/" + training["class"][i] + "/" + training["name"][i],
                    "training/" + training["class"][i] + "/" + training["name"][i])

    for i in range(test.shape[0]):
        shutil.copy("data/" + test["class"][i] + "/" + test["name"][i],
                    "test/" + test["class"][i] + "/" + test["name"][i])

# if len(os.listdir('training_images/')) == 0:
#     files = os.listdir("training/")
#     training = pd.DataFrame()
#     training_list = []
#     training_tags = []
#     for f in files:
#         dir = os.listdir("data/" + f)
#         for dir_f in dir:
#             training_list.append(dir_f)
#             training_tags.append(f)
#
#     training["name"] = training_list
#     training["class"] = training_tags
#
#     for i in range(training.shape[0]):
#         if not os.path.exists("training_images/" + training["class"][i]):
#             Path('training_images/' + training["class"][i]).mkdir(parents=True, exist_ok=True)

if len(os.listdir('training_images/')) == 0:
    files = os.listdir("training/")
    training = pd.DataFrame()
    training_list = []
    training_tags = []
    for f in files:
        dir = os.listdir("data/" + f)
        for dir_f in dir:
            training_list.append(dir_f)
            training_tags.append(f)

    training["name"] = training_list
    training["class"] = training_tags

    path = 'training_images/'
    for i in tqdm(range(training.shape[0])):
        x = 0
        video = training["name"][i]
        cl = training["class"][i]
        cap = cv2.VideoCapture("training/" + cl + "/" + video)

        ret, frame = cap.read()
        while ret:
            filename = video.split(".")[0] + "_frame_" + str(x) + '.jpg'
            fullname = os.path.join(path, filename)
            print("capturing frame " + str(x) + " from video " + video)
            cv2.imwrite(fullname, frame)
            ret, frame = cap.read()
            x += 1



