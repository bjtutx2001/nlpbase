import pandas as pd
from sklearn.model_selection import train_test_split

from src import config
from JiebaTokenizer import BaseTokenizer,EnglishTokenizer,ChineseTokenizer


def process():
    df=pd.read_csv(config.RAW_DIR/"cmn.txt",sep="\t",header=None,usecols=[0,1],names=["en","zh"],encoding="utf-8").dropna()
    print(df.head())

    train_df,test_df=train_test_split(df,test_size=0.3,random_state=37)

    ChineseTokenizer.build_words_list(train_df['zh'],config.PROCESSED_DIR/'zh_unique_words.txt')
    EnglishTokenizer.build_words_list(train_df['en'],config.PROCESSED_DIR/'eh_unique_words.txt')

if __name__=='__main__':
    process()