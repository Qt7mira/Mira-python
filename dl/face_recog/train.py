import numpy as np
import itertools
from scipy.misc import *
import matplotlib.pylab as plt
import os, re
from pprint import pprint
from keras.utils import np_utils
from sklearn.model_selection import train_test_split


def load_image2(path):
    return imread(path)


def load_data():
    x = []
    y = []
    names = []
    data_path = "/Users/panqiutong/Downloads/lfw"
    label = 0
    for folder in os.listdir(data_path):
        folder_path = os.path.join(data_path, folder)

        for file in os.listdir(folder_path):
            img = load_image2(os.path.join(folder_path, file))
            x.append(img)
            y.append(label)

        names.append(folder)
        label += 1

        if label >= 10:
            break
    print("all label", label)
    return x, y, names


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


X, Y, names = load_data()

# with open('model/names.txt', 'a', encoding='utf8') as outfile:
#     for i in names:
#         outfile.write(i)
#         outfile.write('\n')

print("数据加载完成")
X = np.vstack([np.array(list(X))])

print(Y[0:20])
Y = np_utils.to_categorical(Y)
print(Y[0:20])
out_dim = len(Y[0])
print("维数", out_dim)
Y = np.array(Y)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

x_5 = x_train[5]

fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1)
ax1.imshow(x_5[:, :, :])
#
ds = 32
x_train = np.asarray([crop_and_downsample(x, downsample_size=ds) for x in x_train])
x_test = np.asarray([crop_and_downsample(x, downsample_size=ds) for x in x_test])
#
ax2 = fig.add_subplot(1, 2, 2)
ax2.imshow(x_train[5, :, :, :])

plt.show()

print(x_train.shape)
print("数据处理完成")

#
# from keras.models import Sequential
# from keras.layers.core import Dense, Dropout, Activation, Flatten
# from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
# from keras.constraints import maxnorm
# from keras.optimizers import SGD
# from keras.utils import np_utils
#
# model = Sequential()
# model.add(Conv2D(32, (5, 5),
#                   input_shape=(ds, ds, 3),
#                   padding='same',
#                   data_format='channels_last',
#                   activation='relu'))
#
# model.add(Conv2D(32, (5, 5),
#                   padding='same',
#                   data_format='channels_last',
#                   activation='relu'))
#
# model.add(AveragePooling2D(pool_size=(2, 2),
#                             data_format='channels_last'))
#
# model.add(Dropout(0.2))
#
# # Flatten layer.
# model.add(Flatten())
#
# # Fully connected layer with 128 units and a rectifier activation function.
# model.add(Dense(128, activation='relu', kernel_constraint=maxnorm(3)))
#
# # Dropout set to 50%.
# model.add(Dropout(0.5))
#
# # Fully connected output layer with 2 units (Y/N)
# # and a softmax activation function.
# model.add(Dense(out_dim, activation='sigmoid'))
#
# print(model.summary())
#
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#
# epochs = 25
# batch_size = 128
#
# model.fit(x_train, y_train,
#            batch_size=batch_size,
#            epochs=epochs,
#            verbose=1,
#            validation_data=(x_test, y_test))
#
#
# score = model.evaluate(x_test, y_test, verbose=0)
# print('Test accuracy: {0:%}'.format(score[1]))
#
# model.save('model/fr.hdf5')
