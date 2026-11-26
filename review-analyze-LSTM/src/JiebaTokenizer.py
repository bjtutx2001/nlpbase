#类方法提供额外的对象创建方式，类方法通常返回一个对象
import jieba
from datasets import tqdm


class JiebaTokenizer:

    unk_token='<unk>'
    pad_token='<pad>'

    def __init__(self,unique_words):
        self.unique_words = unique_words
        self.unique_words_size=len(unique_words)
        self.word2index={word:index for index,word in enumerate(self.unique_words)}
        self.index2word={index:word for index,word in enumerate(self.unique_words)}
        self.unk_token_index=self.word2index[self.unk_token]
        self.pad_token_index=self.word2index[self.pad_token]

    @staticmethod
    def tokenize(sentence):
        return jieba.lcut(sentence)

    def encode(self,sentence,seq_len):
        tokens = [token for token in self.tokenize(sentence) if token.strip()]
        #截取/填充到指定的长度
        if len(tokens)>seq_len:
            tokens = tokens[:seq_len]
        elif len(tokens)<seq_len:
            tokens=tokens + [self.pad_token]*(seq_len-len(tokens))
        else:
            pass
        return [self.word2index.get(token,self.unk_token_index) for token in tokens]

    @classmethod
    def build_words_list(cls,sentences,path):
        words_set=set()
        for sentence in tqdm(sentences,desc='构建词表'):
            words_set.update(cls.tokenize(sentence))        #update会遍历列表并把元素注意添加到集合
        words_list=[cls.pad_token,cls.unk_token]+sorted(words_set)    #sorted会将传入的可迭代对象进排序，并返回一个新的列表
        words_list=[token for token in words_list if token.strip()]
        print(f'词表大小：{len(words_list)}')

        with open(path,'w',encoding='utf-8') as f:
            f.write('\n'.join(words_list))

    @classmethod
    def from_words(cls,words_path):
        with open(words_path,'r',encoding='utf-8') as f:
            words_list=[line.strip() for line in f.readlines()]

        return cls(words_list)

if __name__ == '__main__':
    pass