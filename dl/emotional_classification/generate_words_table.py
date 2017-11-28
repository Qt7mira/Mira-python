# from __future__ import division
import numpy as np
import pandas as pd
import jieba

all_title = pd.read_excel('data/all_title.xlsx', header=None)
print(len(all_title))

data_all = []
try:
    for line in all_title[0]:
        data_all.append(jieba.lcut(str(line)))
except Exception as e:
    print(e)

all_title['words'] = pd.Series(data_all)

words = {}
for l in all_title['words']:
    for w in l:
        if w in words:
            words[w] += 1
        else:
            words[w] = 1

min_count = 1  # 词频低于min_count的舍弃

words = {i: j for i, j in words.items() if j >= min_count}
id2word = {i + 1: j for i, j in enumerate(words)}  # id映射到词，未登录词全部用0表示
max = len(id2word)
print(max)
word2id = {j: (i-1)/(max-1) for i, j in id2word.items()}  # 词映射到id

print(len(word2id))

# for d, x in word2id.items():
#     print("key:"+d+",value:"+str(x))
import json

with open('model/word2id_50w_norm.json', 'a', encoding='utf8') as outfile:
    json.dump(word2id, outfile)
    outfile.write('\n')
