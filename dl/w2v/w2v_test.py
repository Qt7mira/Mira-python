from gensim.models import Word2Vec
model = Word2Vec.load('model/word2vec_test.m')
print(model.most_similar('科学'))
print(model.most_similar('优势'))
print(model.most_similar('微信'))

# print(model['科学'])
