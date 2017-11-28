import numpy as np
import pandas as pd
import jieba
import json

file = open('model/word2id_1123.json', 'r')
for line in file.readlines():
    word2id = json.loads(line)

maxlen = 50


def doc2num(s):
    s = [word2id.get(i, 0) for i in s[:maxlen]]
    return s + [0] * (maxlen - len(s))


def saliency(s):  # 简单的按saliency排序输出的函数
    ws = jieba.lcut(s)[:maxlen]
    x_ = np.array([[word2id.get(w, 0) for w in ws]])
    score = np.diff(model_dist.predict(x_)[0])
    idxs = score.argsort()
    return [(i, ws[i], -score[i]) for i in idxs]  # 输出结果为：(词位置、词、词权重)

from keras.models import load_model

model = load_model('model/pn_test_all_1127.h5')
model_dist = load_model('model/model_dist_1127.h5')

print(saliency('今天的遭遇简直太糟糕了，真是没谁了，倒霉催的就是。'))
