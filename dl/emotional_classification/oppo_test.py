import numpy as np
import pandas as pd
import jieba

oppo_data = pd.read_excel('data/lenovo.xls', header=None)
print(oppo_data[1].value_counts())
oppo_data['words'] = oppo_data[0].apply(jieba.lcut)

import json
file = open('model/word2id_1123.json', 'r')
for line in file.readlines():
    word2id = json.loads(line)
# word2id = json.load(open('model/word2id_1.json', 'r'))

maxlen = 50

def doc2num(s):
    s = [word2id.get(i, 0) for i in s[:maxlen]]
    return s + [0] * (maxlen - len(s))


oppo_data['id'] = oppo_data['words'].apply(doc2num)

x = np.vstack([np.array(list(oppo_data['id']))])
y = np.array(oppo_data[1])

from keras.models import load_model
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report

model = load_model('model/pn_test_all_1123.h5')

# 准确率与召回率
precision, recall, thresholds = precision_recall_curve(y, model.predict(x))
answer = model.predict(x)
# for i in range(len(answer)):
#     print(answer[i])

report = answer > 0.5
print(classification_report(y, report, target_names=['neg', 'pos']))
