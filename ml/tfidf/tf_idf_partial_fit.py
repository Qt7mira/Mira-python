import re
import numpy as np
from scipy.sparse.dia import dia_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


# TfidfVectorizer的partial_fit方法，但是还没
def partial_fit(self, x):
    max_idx = max(self.vocabulary_.values())
    for a in x:
        # update vocabulary_
        if self.lowercase:
            a = a.lower()
        tokens = re.findall(self.token_pattern, a)
        for w in tokens:
            if w not in self.vocabulary_:
                max_idx += 1
                self.vocabulary_[w] = max_idx

        # update idf_
        df = (self.n_docs + self.smooth_idf)/np.exp(self.idf_ - 1) - self.smooth_idf
        self.n_docs += 1
        df.resize(len(self.vocabulary_))
        for w in tokens:
            df[self.vocabulary_[w]] += 1
        idf = np.log((self.n_docs + self.smooth_idf)/(df + self.smooth_idf)) + 1
        self._tfidf._idf_diag = dia_matrix((idf, 0), shape=(len(idf), len(idf)))
        # print((len(idf), len(idf)))
        # print(vec._tfidf._idf_diag.shape)


TfidfVectorizer.partial_fit = partial_fit
vec = TfidfVectorizer()
articleList = ['here is some text blah blah', 'another text object', 'more foo for your bar right now']
vec.fit(articleList)
vec.n_docs = len(articleList)
vec.partial_fit(['the last text I wanted to add'])
print(vec.transform(['the last text I wanted to add']).toarray())
print(vec.get_feature_names())
