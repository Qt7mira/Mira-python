from Word2Vec_nk import *
import pymongo
import jieba


class Sentences:  # 语料生成器，必须这样写才是可重复使用的
    def __init__(self):
        self.db = pymongo.MongoClient('mongodb://root:112358s@182.92.196.59 ', 27017).usearch.v4_articles_171103

    def __iter__(self):
        for t in self.db.find({}, {'title': '1', 'content': '1'}, no_cursor_timeout=True).limit(10000):
            # tit_con = ""
            # for key, value in t.items():
            #     if value is None:
            #         tit_con += ""
            #     else:
            #         tit_con += value
            # yield list(jieba.cut(tit_con))  # 返回分词后的结果
            yield t['title']

wv = Word2Vec_nk(Sentences(), model='cbow', nb_negative=16, shared_softmax=True, epochs=2) # 建立并训练模型
wv.save_model('myvec')  # 保存到当前目录下的myvec文件夹

# 训练完成后可以这样调用
wv = Word2Vec_nk()  # 建立空模型
wv.load_model('myvec')  # 从当前目录下的myvec文件夹加载模型
