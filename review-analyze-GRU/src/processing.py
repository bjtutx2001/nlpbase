import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from JiebaTokenizer import JiebaTokenizer
import config



def process():
    print('开始处理数据')

    #读取数据
    df=pd.read_csv(config.RAW_DIR / 'online_shopping_10_cats.csv',usecols=['label','review'],encoding='utf-8').dropna().sample(frac=0.1)
    # df=pd.read_csv(config.RAW_DIR / 'online_shopping_10_cats.csv',usecols=['label','review'],encoding='utf-8').dropna()
    df.info()

    #划分数据集
    train_df,test_df=train_test_split(df,test_size=0.3,random_state=37,stratify=df['label'])

    #构建词表
    JiebaTokenizer.build_words_list(train_df['review'].tolist(),config.PROCESSED_DIR / 'unique_words.txt')
    tokenizer=JiebaTokenizer.from_words(config.PROCESSED_DIR / 'unique_words.txt')

    #计算序列长度
    # quantile()是Pandas中用于计算 ** 分位数（Quantile） ** 的方法，适用于Series和DataFrame对象。
    train_df['review'].apply(lambda x: len(tokenizer.tokenize(x))).quantile(0.95)

    #构建训练集
    train_df['review']=train_df['review'].apply(lambda x:tokenizer.encode(x,config.SEQ_LEN))
    #保存训练集
    train_df.to_json(config.PROCESSED_DIR / 'train.jsonl',orient='records',lines=True)

    #创建测试集
    test_df['review'] = test_df['review'].apply(lambda x: tokenizer.encode(x,config.SEQ_LEN))
    test_df.to_json(config.PROCESSED_DIR / 'test.jsonl', orient='records', lines=True)

    print('数据处理完成')


if __name__=='__main__':
    process()