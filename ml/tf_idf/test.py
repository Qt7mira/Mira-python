str1 = '在浏览Keras官方文档时'
windows = 10
import random


def train_batch_sg(sentences):
    result = 0
    for sentence in sentences:
        word_vocabs = sentence
        for pos, word in enumerate(word_vocabs):
            reduced_window = random.randint(1, windows)
            start = max(0, pos - windows + reduced_window)
            print('---------------------------------------------')
            print('ssss:' + str(pos) + " " + str(word))
            for pos2, word2 in enumerate(word_vocabs[start:(pos + windows + 1 - reduced_window)], start):
                if pos2 != pos:
                    print(str(pos2) + " " + str(word2))
                    print('do train_sg_pair')
            print('--------------done this cycle---------------')
        result += len(word_vocabs)
    return result


import jieba
print(jieba.lcut(str1))
train_batch_sg(jieba.lcut(str1))
