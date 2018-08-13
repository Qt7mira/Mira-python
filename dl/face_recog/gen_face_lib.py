from scipy.misc import *
import os
import datetime
from dl.face_recog.model import FrModel
import numpy as np


def load_image2(path):
    return imread(path)


def load_data():
    x = []
    y = []
    data_path = "/Users/panqiutong/Downloads/lfw"
    label = 0
    for folder in os.listdir(data_path):
        # folder_path = os.path.join(data_path, folder, folder + "_0001.jpg")
        # img = load_image2(folder_path)
        # x.append(img)
        y.append(folder)
        label += 1

        # if label >= 10:
        #     break

    return x, y


def calc_cos(vec_1, vec_2):
    num = np.dot(vec_1, vec_2)
    denom = np.linalg.norm(vec_1) * np.linalg.norm(vec_2)
    cos = num / denom
    return cos


def calc_o(vec_1, vec_2):
    dist = np.linalg.norm(vec_1 - vec_2)
    return dist


x, y = load_data()
fm = FrModel()
x = [fm.img_2_vec(i) for i in x]

# time_point_1 = datetime.datetime.now()
# path_2 = "/Users/panqiutong/Downloads/lfw/George_W_Bush/George_W_Bush_0008.jpg"
# vec_2 = fm.img_path_2_vec(path_2)
# time_point_2 = datetime.datetime.now()
# print("转向量", str(time_point_2 - time_point_1))
#
# result = [calc_cos(i, vec_2) for i in x]
# min_r = max(result)
# print(min_r)
# who = y[result.index(min_r)]
# print(who)
# time_point_3 = datetime.datetime.now()
# print("计算相似度", str(time_point_3 - time_point_2))
# print("总时长", str(time_point_3 - time_point_1))


time_point_1 = datetime.datetime.now()
# path_2 = "/Users/panqiutong/Downloads/lfw/George_W_Bush/George_W_Bush_0008.jpg"
path_2 = "/Users/panqiutong/Downloads/lfw/Paul_Wilson/Paul_Wilson_0001.jpg"
img2 = imread(path_2)
result = fm.predict(img2)
print(result)
print(np.max(result))
location = np.where(result == np.max(result))[1][0]
print(location)
print(y.index("Paul_Wilson"))
who = y[location]
print(who)
time_point_3 = datetime.datetime.now()
print("总时长", str(time_point_3 - time_point_1))
