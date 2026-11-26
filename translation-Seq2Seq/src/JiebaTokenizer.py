#类方法提供额外的对象创建方式，类方法通常返回一个对象
import jieba
from datasets import tqdm
from nltk import TreebankWordTokenizer, TreebankWordDetokenizer


class BaseTokenizer:

    unk_token='<unk>'
    pad_token='<pad>'
    sos_token='<sos>'
    eos_token='<eos>'

    def __init__(self,unique_words):
        self.unique_words = unique_words
        self.unique_words_size=len(unique_words)
        self.word2index={word:index for index,word in enumerate(self.unique_words)}
        self.index2word={index:word for index,word in enumerate(self.unique_words)}
        self.unk_token_index=self.word2index[self.unk_token]
        self.pad_token_index=self.word2index[self.pad_token]
        self.eos_token_index=self.word2index[self.eos_token]
        self.sos_token_index=self.word2index[self.sos_token]

    @classmethod
    def tokenize(cls,sentence)->list[str]:      #声明返回值
        pass

    def encode(self,sentence):
        tokens = [token for token in self.tokenize(sentence) if token.strip()]
        return [self.word2index.get(token,self.unk_token_index) for token in tokens]

    @classmethod
    def build_words_list(cls,sentences,path):
        words_set=set()
        for sentence in tqdm(sentences,desc='构建词表'):
            words_set.update(cls.tokenize(sentence))        #update会遍历列表并把元素注意添加到集合
        words_list=[cls.pad_token,cls.unk_token,cls.sos_token,cls.eos_token]+sorted(words_set)    #sorted会将传入的可迭代对象进排序，并返回一个新的列表
        words_list=[token for token in words_list if token.strip()]
        print(f'词表大小：{len(words_list)}')

        with open(path,'w',encoding='utf-8') as f:
            f.write('\n'.join(words_list))

    @classmethod
    def from_words(cls,words_path):
        with open(words_path,'r',encoding='utf-8') as f:
            words_list=[line.strip() for line in f.readlines()]

        return cls(words_list)

class ChineseTokenizer(BaseTokenizer):
    # def __init__(self,unique_words):
    #     super().__init__(unique_words)

    @classmethod
    def tokenize(cls,sentence)->list[str]:
        return list(sentence)


class EnglishTokenizer(BaseTokenizer):
    tokenizer=TreebankWordTokenizer()
    detokenizer=TreebankWordDetokenizer()

    @classmethod
    def tokenize(cls,sentence)->list[str]:
        return cls.tokenizer.tokenize(sentence)

    def decode(self,indexes):
        tokens =[self.index2word[index] for index in indexes]
        return self.detokenizer.detokenize(tokens)




if __name__ == '__main__':
    pass