import jieba
import pandas as pd
from pathlib import Path

from datasets import tqdm
from sklearn.model_selection import train_test_split

import config

def process_data():
    print("开始处理数据")
    print(config.RAW_DATA_DIR)
    #1、读取文件
    df=pd.read_json(config.RAW_DATA_DIR /"synthesized_.jsonl", lines=True, orient='records').sample(frac=0.1,random_state=37)   #sample方法对数据进行抽样，在测试的时候用，加快数据加载进度，frac为比例
    print(df.head())

    #2、读取所有句子
    sentences=[]
    # print(type(df['dialog']))
    for sentence in df['dialog']:
        # print(type(sentence))     结果是list
        for ju in sentence:
            sentences.append(ju.split('：')[1])      #用：把user1和具体的评论分开，取评论作为句子
    #查看提取出的评论
    # print(sentences)
    # print(len(sentences))

    #3、划分数据集
    train_sentences,test_sentences=train_test_split(sentences,test_size=0.2,random_state=37)

    #4、基于训练集构建词表
    unique_words=[]
    for sentence in tqdm(train_sentences,desc='构建词表'):
        for word in jieba.lcut(sentence):
            if word not in unique_words:
                unique_words.append(word)
    unique_words=['<unk>']+unique_words
    print(len(unique_words))

    #5、保存词表与词表对应的索引
    with open(config.MODELS_DIR / 'unique_words.txt', 'w',encoding='utf-8') as f:
        f.write('\n'.join(unique_words))    #jion:使用指定的分隔符，将多个字符串拼接成一个完整的字符串。必须是可迭代对象

    word2index={index:word for index,word in enumerate(unique_words)}

    print('数据处理完成')


if __name__ == '__main__':
    process_data()