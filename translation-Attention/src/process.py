import pandas as pd
from sklearn.model_selection import train_test_split

from src import config
from JiebaTokenizer import BaseTokenizer,EnglishTokenizer,ChineseTokenizer


def process():
    df=pd.read_csv(config.RAW_DIR/"cmn.txt",sep="\t",header=None,usecols=[0,1],names=["en","zh"],encoding="utf-8").dropna(how="all")
    # print(df.head())

    train_df,test_df=train_test_split(df,test_size=0.2,random_state=37)

    #构建词表
    ChineseTokenizer.build_words_list(train_df['zh'],config.PROCESSED_DIR/'zh_unique_words.txt')
    EnglishTokenizer.build_words_list(train_df['en'],config.PROCESSED_DIR/'en_unique_words.txt')

    #创建tokenizer对象
    zh_tokenizer=ChineseTokenizer.from_words(config.PROCESSED_DIR/'zh_unique_words.txt')
    en_tokenizer=EnglishTokenizer.from_words(config.PROCESSED_DIR/'en_unique_words.txt')

    #构建训练集
    train_df['zh']=train_df['zh'].apply(lambda x: zh_tokenizer.encode(x,add_sos_eos=False))
    train_df['en']=train_df['en'].apply(lambda x: en_tokenizer.encode(x,add_sos_eos=True))

    #保存训练接
    train_df.to_json(config.PROCESSED_DIR/"train.jsonl",orient="records",lines=True)

    #构建并保存测试集
    test_df['zh']=test_df['zh'].apply(lambda x: zh_tokenizer.encode(x,add_sos_eos=False))
    test_df['en']=test_df['en'].apply(lambda x: en_tokenizer.encode(x,add_sos_eos=True))
    test_df.to_json(config.PROCESSED_DIR / "test.jsonl", orient="records", lines=True)

    print(train_df.head())

if __name__=='__main__':
    process()