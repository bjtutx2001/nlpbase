"""
通过Gensim库，基于公开的语料库，训练自己的词向量
训练好的词向量可以用于初始化下游NLP任务的词嵌入层，并且加载好词向量后，可以选择是否根据反向传播更新词向量
"""
import pandas as pd
import jieba
from gensim.models import KeyedVectors, Word2Vec

review=pd.read_csv('./data/1.评论数据集/online_shopping_10_cats.csv', encoding='utf-8',sep=',')
#取出含有无效内容的行
review=review.dropna()
print(review.head())
review.info()
#切词并且去除空格和转义字符
sen=[[token.strip() for token in jieba.lcut(senten) if token.strip()] for senten in review['review']]
# print(sen[0:3])

#用gensim工具训练词向量
model=Word2Vec(
    sentences=sen,      #传入已分词的序列
    vector_size=100,    #词向量维度
    window=5,           #上下文窗口大小
    min_count=5,        #最小词频（小于最小词频的词舍弃）
    sg=1,               #Skip-Gram：1，CBOW：0
    workers=4,          #并行线程数
)

print(model.wv.vector_size)
print(model.wv[10])

#保存词向量
model.wv.save_word2vec_format('./model/自训练词向量.txt')
