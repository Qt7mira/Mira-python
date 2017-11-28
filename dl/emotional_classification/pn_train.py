import numpy as np
import pandas as pd
import jieba


data = pd.read_excel('data/newdata1123.xlsx', header=None)
print(len(data))

data_cut = []
try:
    for line in data[0]:
        data_cut.append(jieba.lcut(str(line)))
except Exception as e:
    print(e)

data['words'] = pd.Series(data_cut)

# words = {}
# for l in data['words']:
#     for w in l:
#         if w in words:
#             words[w] += 1
#         else:
#             words[w] = 1


maxlen = 50  # 句子截断为100字

# min_count = 1  # 词频低于min_count的舍弃
# words = {i: j for i, j in words.items() if j >= min_count}
# id2word = {i + 1: j for i, j in enumerate(words)}  # id映射到词，未登录词全部用0表示
# word2id = {j: i for i, j in id2word.items()}  # 词映射到id

import json
file = open('model/word2id_1123.json', 'r')
for line in file.readlines():
    word2id = json.loads(line)

print(len(word2id))

# for d, x in word2id.items():
#     print("key:"+d+",value:"+str(x))
# import json

# with open('model/word2id_1123.json', 'a', encoding='utf8') as outfile:
#     json.dump(word2id, outfile)
#     outfile.write('\n')


def doc2num(s):
    s = [word2id.get(i, 0) for i in s[:maxlen]]
    return s + [0] * (maxlen - len(s))


data['id'] = data['words'].apply(doc2num)

x = np.vstack([np.array(list(data['id']))])
y = np.array(data[1])

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.11)

from keras.models import Model
from keras.layers import Input, Dense, Dropout, Embedding, Lambda
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import LSTM
from keras import regularizers
from keras import backend as K

# 建立模型
input = Input(shape=(None,))
input_vecs = Embedding(len(word2id) + 1, 128, mask_zero=True)(input)  # 用了mask_zero，填充部分自动为0
# input_vecs = Embedding(len(word2id) + 1, 128)(input)
# conv = Conv1D(activation='relu', padding='same', filters=200, kernel_size=5, kernel_regularizer=regularizers.l2(0.01))(input_vecs)
# pooling = MaxPooling1D(pool_size=2)(conv)
# dropout0 = Dropout(0.25)(pooling)
lstm = LSTM(128, return_sequences=True, return_state=True)(input_vecs)  # 返回一个list
lstm_state = Lambda(lambda x: x[1])(lstm)  # list的第二个元素就是lstm最后的状态
dropout = Dropout(0.5)(lstm_state)
predict = Dense(1, activation='sigmoid')(dropout)

# list的第一个元素就是lstm的状态向量序列，先补充一个0向量（h_0），然后与
lstm_sequence = Lambda(lambda x: K.concatenate([K.zeros_like(x[0])[:, :1], x[0]], 1))(lstm)
lstm_dist = Lambda(
    lambda x: K.sqrt(K.sum((x[0] - K.expand_dims(x[1], 1)) ** 2, 2) / K.sum(x[1] ** 2, 1, keepdims=True)))(
    [lstm_sequence, lstm_state])

model = Model(inputs=input, outputs=predict)  # 文本情感分类模型
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model_dist = Model(inputs=input, outputs=lstm_dist)  # 计算权重的模型
model_dist.compile(loss='mse',
                   optimizer='adam')

batch_size = 128

model.fit(x_train, y_train, batch_size=batch_size, epochs=10, validation_data=(x_test, y_test))
model.save('model/pn_test_all_1127.h5')
model_dist.save('model/model_dist_1127.h5')

# from keras.utils import plot_model
# plot_model(model, to_file='model.png')


def saliency(s):  # 简单的按saliency排序输出的函数
    ws = jieba.lcut(s)[:maxlen]
    x_ = np.array([[word2id.get(w, 0) for w in ws]])
    score = np.diff(model_dist.predict(x_)[0])
    idxs = score.argsort()
    return [(i, ws[i], -score[i]) for i in idxs]  # 输出结果为：(词位置、词、词权重)


print(model.evaluate(x_test, y_test, batch_size=batch_size))

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report

# 准确率与召回率
precision, recall, thresholds = precision_recall_curve(y_test, model.predict(x_test))
answer = model.predict(x_test)
report = answer > 0.5
print(classification_report(y_test, report, target_names=['neg', 'pos']))

print(saliency('今天的遭遇简直太糟糕了，真是没谁了，倒霉催的就是。'))
