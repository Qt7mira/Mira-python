from scipy.misc import *
import os
import datetime
from dl.face_recog.model import FrModel
import numpy as np
import pandas as pd


def load_image2(path):
    return imread(path)


# def load_data():
#     x = []
#     y = []
#     data_path = "/Users/panqiutong/Downloads/lfw"
#     label = 0
#     for folder in os.listdir(data_path):
#         # folder_path = os.path.join(data_path, folder, folder + "_0001.jpg")
#         # img = load_image2(folder_path)
#         # x.append(img)
#         y.append(folder)
#         label += 1
#
#         print(folder)
#         if label >= 150:
#             break
#
#     return x, y


def load_data():
    x = []
    y = pd.read_csv("model/names.txt")['name'].tolist()

    data_path = "/Users/panqiutong/Downloads/lfw"
    for name in y:
        folder_path = os.path.join(data_path, name, name + "_0001.jpg")
        img = load_image2(folder_path)
        x.append(img)
    # print(y[0:50])
    return x, y


def calc_cos(vec_1, vec_2):
    num = np.dot(vec_1, vec_2)
    denom = np.linalg.norm(vec_1) * np.linalg.norm(vec_2)
    cos = num / denom
    return cos


def calc_o(vec_1, vec_2):
    dist = np.linalg.norm(vec_1 - vec_2)
    return dist


# x, y = load_data()
print("数据加载完成")
fm = FrModel()

path = "/Users/panqiutong/Downloads/lfw/George_W_Bush/George_W_Bush_0001.jpg"
vec_1 = fm.img_path_2_vec(path)

m = 1
for i in range(2, 6):
    path = "/Users/panqiutong/Downloads/lfw/George_W_Bush/George_W_Bush_000" + str(i) + ".jpg"
    vec_1 += fm.img_path_2_vec(path)
    m += 1
vec_1 = vec_1 / m


time_point_1 = datetime.datetime.now()
path_2 = "/Users/panqiutong/Downloads/lfw/George_W_Bush/George_W_Bush_0008.jpg"
vec_2 = fm.img_path_2_vec(path_2)
print(vec_2)
time_point_2 = datetime.datetime.now()
print("转向量", str(time_point_2 - time_point_1))

result = [calc_cos(vec_1, vec_2)]
print(result)
time_point_3 = datetime.datetime.now()
print("计算相似度", str(time_point_3 - time_point_2))
print("总时长", str(time_point_3 - time_point_1))

