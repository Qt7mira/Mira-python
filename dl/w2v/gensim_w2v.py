import gensim
import logging
import pymongo
import jieba
import xlrd


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class Sentences:
    def __init__(self):
        self.db = pymongo.MongoClient('mongodb://root:112358s@182.92.196.59 ', 27017).usearch.v4_articles_171115

    def __iter__(self):
        # for t in self.db.find({}, {'title': '1', 'content': 1}, no_cursor_timeout=True).limit(10000):
        #     tit_con = ""
        #     for key, value in t.items():
        #         if value is None:
        #             tit_con += ""
        #         else:
        #             tit_con += value
        #     yield list(jieba.cut(tit_con))  # 返回分词后的结果
        datafile = xlrd.open_workbook('C:/Users/Administrator/Desktop/test.xlsx')
        table = datafile.sheet_by_name('Sheet1')
        num_rows = table.nrows
        for row in range(0, num_rows):
            yield list(jieba.cut(str(table.cell(row, 0).value).replace('\n', '')))

word2vec = gensim.models.word2vec.Word2Vec(Sentences(), size=256, window=10, min_count=10, sg=1, hs=1, iter=10, workers=25)
word2vec.save('word2vec_test.m')


print(word2vec.most_similar('科学'))
print(word2vec.most_similar('交通'))
