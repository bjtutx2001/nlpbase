import jieba
from datasets import tqdm

class Jiebatokenizer:
    unk_token='<unk>'
    def __init__(self,words_list):
        self.words_list=words_list
        self.words_size=len(words_list)
        self.word2index={word:index for index,word in enumerate(words_list)}
        self.index2word={index:word for index,word in enumerate(words_list)}
        self.unk_token_index=self.word2index[self.unk_token]

    @staticmethod   #静态方法，不依赖类属性和实例属性，只是与类相关
    def tokenize(text):
        return jieba.lcut(text)

    def encode(self,text):      #实例方法，对输入的句子按照唯一词表映射成索引
        tokens=self.tokenize(text)
        return [self.word2index.get(token,self.unk_token_index) for token in tokens]

    @classmethod    #类方法
    def bulid_words(cls,sentences,path):
        words_set=set()
        for sentence in tqdm(sentences,desc='构建词表'):
            words_set.update(jieba.lcut(sentence))

        words_list=[cls.unk_token] + list(words_set)

        with open(path,'w',encoding='utf-8') as f:
            f.write('\n'.join(words_list))

    @classmethod
    def from_words(cls,words_path):
        with open(words_path,'r',encoding='utf-8') as f:
            words_list=f.read().splitlines()

        return cls(words_list)  #使用 words_list 作为参数创建当前类的实例对象，并将该实例对象返回。等价于Jiebatokenizer(words_list)创建实例对象

if __name__=='__main__':
    pass