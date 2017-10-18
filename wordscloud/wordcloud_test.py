import jieba
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from wordcloud import WordCloud


from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False


def wordcloudplot(txt):
    # 加载字体
    path = 'data/msyh.ttf'
    # path = unicode(path, 'utf8').encode('gb18030')
    alice_mask = np.array(Image.open('img/heart.jpg'))
    wc = WordCloud(font_path=path,
                   background_color="white",
                   margin=5, width=1800, height=800, mask=alice_mask, max_words=2000, max_font_size=60,
                   random_state=42)
    word_cloud = wc.generate(txt)
    # word_cloud = wc.fit_words(txt)
    word_cloud.to_file('img/wordcloud_test.jpg')
    plt.imshow(word_cloud)
    plt.axis("off")
    plt.show()


def main():
    a = []
    doc = ""
    with open('data/gdcs.txt', 'r', encoding='utf-8') as f:
        for line in f:
            doc += line.strip()

    words = list(jieba.cut(doc))
    for word in words:
        if len(word) > 1:
            a.append(word)
    txt = ' '.join(a)
    # print(txt)
    wordcloudplot(txt)


if __name__ == '__main__':
    main()
