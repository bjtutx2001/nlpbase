import jieba
import pandas as pd
from pathlib import Path

from datasets import tqdm
from sklearn.model_selection import train_test_split

import config

#创建函数，表示构建数据集
def bulid_Dataset(data,word2index):
    dataset=[]
    #6、构建训练集的特征与标签
    for sentence in tqdm(data,desc='构建数据集'):
        #获取分词结果对应的词索引
        word_index=[word2index.get(word,0) for word in jieba.lcut(sentence)]
        # print(word_index)
        for i in range(len(jieba.lcut(sentence))-config.seq_len):
            x=word_index[i:i+config.seq_len]
            y=word_index[i+1:i+config.seq_len+1]
            dataset.append({'x':x,'y':y})
    return dataset

def process_data():
    print("开始处理数据")
    print(config.RAW_DATA_DIR)
    #1、读取文件
    # df=pd.read_json(config.RAW_DATA_DIR /"synthesized_.jsonl", lines=True, orient='records').sample(frac=0.01,random_state=37)   #sample方法对数据进行抽样，在测试的时候用，加快数据加载进度，frac为比例
    df=pd.read_json(config.RAW_DATA_DIR /"synthesized_.jsonl", lines=True, orient='records')
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
    unique_words=['<unk>']+unique_words     #添加未知字符
    print(len(unique_words))

    #5、保存词表与词表对应的索引
    with open(config.MODELS_DIR / 'unique_words.txt', 'w',encoding='utf-8') as f:
        f.write('\n'.join(unique_words))    #jion:使用指定的分隔符，将多个字符串拼接成一个完整的字符串。必须是可迭代对象

    word2index={word:index for index,word in enumerate(unique_words)}

    # print(pd.Series(word2index))                                                      #保留原始中文，设置缩进为4，会自动换行
    pd.Series(word2index).to_json(config.MODELS_DIR / 'word2index.json',orient='index',force_ascii=False,indent=4)

    # train_dataset=[]
    # #6、构建训练集的特征与标签
    # for sentence in train_sentences:
    #     #获取分词结果对应的词索引
    #     word_index=[word2index.get(word,0) for word in jieba.lcut(sentence)]
    #     # print(word_index)
    #     for i in range(len(jieba.lcut(sentence))-config.seq_len):
    #         x_train=word_index[i:i+config.seq_len]
    #         y_train=word_index[i+config.seq_len]
    #         train_dataset.append({'x_train':x_train,'y_train':y_train})
    # print(train_dataset[90:111])

    train_dataset=bulid_Dataset(train_sentences,word2index)
    #保存训练集

    pd.DataFrame(train_dataset).to_json(config.PROCESSED_DATA_DIR / 'train_dataset.jsonl',orient='records',lines=True,force_ascii=False)

    #7、构建测试集
    test_dataset=bulid_Dataset(test_sentences,word2index)
    pd.DataFrame(test_dataset).to_json(config.PROCESSED_DATA_DIR / 'test_dataset.jsonl',orient='records',lines=True,force_ascii=False)

    print('数据处理完成')


if __name__ == '__main__':
    process_data()