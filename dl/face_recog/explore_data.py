# numerics
import numpy as np
import itertools

# images
from scipy.misc import *
#imresize, imread, imshow
import matplotlib.pylab as plt

# dealing with tar files
import tarfile


def load_image(tgz_file, basename, name, number):

    filename = "{0}/{1}/{1}_{2:04d}.jpg".format(basename, name, int(number))
    tgz = tarfile.open(tgz_file)
    return imread(tgz.extractfile(filename))


def load_image2(path, basename, name, number):
    filename = path + "{0}/{1}/{1}_{2:04d}.jpg".format(basename, name, int(number))
    print(filename)
    return imread(filename)

# Load image:
# tgz = "/Users/panqiutong/Downloads/lfw.tgz"
# z = load_image(tgz, "lfw", "George_W_Bush", 5)
# print("Shape of image: W x H x RGB")
# print(np.shape(z))


path = "/Users/panqiutong/Downloads/"
z = load_image2(path, "lfw", "George_W_Bush", 5)
print("Shape of image: W x H x RGB")
print(np.shape(z))

# plt.imshow(np.uint8(z))
# plt.show()
#
# fig = plt.figure(figsize=(14,6))
# [ax1, ax2, ax3] = [fig.add_subplot(1,3,i+1) for i in range(3)]
#
# ax1.imshow(z[:,:,0],cmap="gray")
# ax2.imshow(z[:,:,1],cmap="gray")
# ax3.imshow(z[:,:,2],cmap="gray")
# plt.show()


def extract_features(z):
    features = np.array([z[:, :, 0], z[:, :, 1], z[:, :, 2]])
    return features


features = extract_features(z)
print(np.shape(features))

