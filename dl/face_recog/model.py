
from scipy.misc import *

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, AveragePooling2D
from keras.constraints import maxnorm
import numpy as np
import tensorflow as tf


def load_image2(path):
    return imread(path)


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
    # model.add(Dense(5749, activation='sigmoid'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def create_model2():
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

    # model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_constraint=maxnorm(3)))
    # model.add(Dropout(0.2))
    model.add(Dense(5749))
    # model.add(Dense(5749, activation='sigmoid'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


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


class FrModel(object):

    def __init__(self):
        self.model = create_model2()
        self.model.load_weights("model/fr_2.hdf5", by_name=True)
        print("模型加载权重完成")

    def predict(self, img):
        X = []
        X.append(img)
        X = np.vstack([np.array(list(X))])
        X = np.asarray([crop_and_downsample(x) for x in X])
        return self.model.predict(X)

    def img_2_vec(self, img):
        X = []
        X.append(img)
        X = np.vstack([np.array(list(X))])
        X = np.asarray([crop_and_downsample(x) for x in X])
        vec = self.model.predict(X)
        return vec[0]

    def img_path_2_vec(self, path):
        X = []
        X.append(load_image2(path))
        X = np.vstack([np.array(list(X))])
        X = np.asarray([crop_and_downsample(x) for x in X])
        vec = self.model.predict(X)
        return vec[0]













# print("计算相似度")
# from numpy import linalg
#
#
# def calc_cos(vec_1, vec_2):
#     num = np.dot(vec_1, vec_2)
#     denom = linalg.norm(vec_1) * linalg.norm(vec_2)
#     cos = num / denom
#     return cos
#
#
# def calc_o(vec_1, vec_2):
#     dist = linalg.norm(vec_1 - vec_2)
#     return dist
#
#
# print("cos", calc_cos(vec_1, vec_2))
# print("cos", calc_cos(vec_1, vec_3))
#
# print("dist", calc_o(vec_1, vec_2))
# print("dist", calc_o(vec_1, vec_3))
# dist = linalg.norm(vec_1 - vec_2)
# print("o", dist)
