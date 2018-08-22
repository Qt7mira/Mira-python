import numpy as np
import itertools
from scipy.misc import *
import matplotlib.pylab as plt
import os, re
from pprint import pprint
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.utils import np_utils
import numpy as np


def load_image2(path):
    return imread(path)


def load_data():
    x = []
    y = []
    data_path = "/Users/panqiutong/Downloads/lfw"
    label = 0
    for folder in os.listdir(data_path):
        folder_path = os.path.join(data_path, folder)

        for file in os.listdir(folder_path):
            img = load_image2(os.path.join(folder_path, file))
            x.append(img)
            y.append(label)
        label += 1

        # if label >= 10:
        #     break
    print("all label", label)
    return x, y


def crop_and_downsample(originalX, downsample_size=32):
    """
    Starts with a 250 x 250 image.
    Crops to 128 x 128 around the center.
    Downsamples the image to (downsample_size) x (downsample_size).
    Returns an image with dimensions (channel, width, height).
    """
    current_dim = 250
    target_dim = 128
    margin = int((current_dim - target_dim) / 2)
    left_margin = margin
    right_margin = current_dim - margin

    # newim is shape (6, 128, 128)
    newim = originalX[left_margin:right_margin, left_margin:right_margin, :]

    # resized are shape (feature_width, feature_height, 3)
    feature_width = feature_height = downsample_size
    newX = imresize(newim[:, :, :], (feature_width, feature_height), interp="bicubic", mode="RGB")

    # the next line is EXTREMELY important.
    # if you don't normalize your data, all predictions will be 0 forever.
    newX = newX / 255

    return newX


def create_model():
    model = Sequential()
    model.add(Conv2D(32, (5, 5),
                     input_shape=(32, 32, 3),
                     padding='same',
                     data_format='channels_last',
                     activation='relu'))

    model.add(Conv2D(32, (5, 5),
                     padding='same',
                     data_format='channels_last',
                     activation='relu'))

    model.add(AveragePooling2D(pool_size=(2, 2),
                               data_format='channels_last'))

    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


model = create_model()
model.load_weights("model/fr_1.hdf5", by_name=True)


def img2vec(path):
    X = []
    X.append(load_image2(path))
    X = np.vstack([np.array(list(X))])
    X = np.asarray([crop_and_downsample(x) for x in X])
    vec = model.predict(X)
    return vec[0]

    # X = load_image2(path)
    # X = np.asarray(crop_and_downsample(X))
    # vec = model.predict(X)
    # return vec


path_1 = "/Users/panqiutong/Downloads/lfw/George_W_Bush/George_W_Bush_0005.jpg"
vec_1 = img2vec(path_1)

path_2 = "/Users/panqiutong/Downloads/lfw/George_W_Bush/George_W_Bush_0003.jpg"
vec_2 = img2vec(path_2)

path_3 = "/Users/panqiutong/Downloads/lfw/Zhu_Rongji/Zhu_Rongji_0003.jpg"
vec_3 = img2vec(path_3)


print("计算相似度")
from numpy import linalg


def calc_cos(vec_1, vec_2):
    num = np.dot(vec_1, vec_2)
    denom = linalg.norm(vec_1) * linalg.norm(vec_2)
    cos = num / denom
    return cos


def calc_o(vec_1, vec_2):
    dist = linalg.norm(vec_1 - vec_2)
    return dist


print("cos", calc_cos(vec_1, vec_2))
print("cos", calc_cos(vec_1, vec_3))

print("dist", calc_o(vec_1, vec_2))
print("dist", calc_o(vec_1, vec_3))
# dist = linalg.norm(vec_1 - vec_2)
# print("o", dist)


