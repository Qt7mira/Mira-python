import numpy as np
import jieba
from PIL import Image
from pylab import mpl
from wordcloud import WordCloud


# mpl.rcParams['font.sans-serif'] = ['SimHei']
# mpl.rcParams['axes.unicode_minus'] = False


# 读取数据，返回分词后的的文本字符串
def read_book(file_path):
    a = []
    doc = ""
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            doc += line.strip()

    # jieba.add_word('路明非')
    # jieba.add_word('楚子航')
    # jieba.add_word('路鸣泽')
    # jieba.add_word('夏弥')
    words = list(jieba.cut(doc))
    for word in words:
        if len(word) > 1:  # 舍去单个字的词
            a.append(word)
    return ' '.join(a)


# 作图
def word_cloud_plot(font_path, image_path, out_path, txt):
    alice_mask = np.array(Image.open(image_path))
    wc = WordCloud(font_path=font_path,
                   background_color="white",
                   margin=0, width=1800, height=800, mask=alice_mask, max_words=2000, max_font_size=40,
                   min_font_size=20,
                   random_state=42)
    word_cloud = wc.generate(txt)
    word_cloud.to_file(out_path)

    # 使用matplotlib，先看看图片效果，程序运行正常时，注释即可
    # import matplotlib.pyplot as plt
    # plt.imshow(word_cloud)
    # plt.axis("off")
    # plt.show()


def main():
    font_path = 'data/msyh.ttf'
    image_path, out_path = 'img/timg2.jpg', 'img/gg.jpg'
    txt = read_book('data/zhushen.txt')
    word_cloud_plot(font_path, image_path, out_path, txt)


if __name__ == '__main__':
    main()
