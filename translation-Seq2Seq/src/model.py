from torch import nn

from src import config


class TranslationEncoder(nn.Module):
    def __init__(self,unique_size,padding_index):
        super().__init__()

        self.embed=nn.Embedding(num_embeddings=unique_size,
                                embedding_dim=config.EMBEDDING_DIM,
                                padding_idx=padding_index           #padding_idx指定哪个词语索引代表填充符 <pad>，使该索引对应的词向量不参与梯度更新。
                                )
        self.gru=nn.GRU(input_size=config.EMBEDDING_DIM,
                        hidden_size=config.HIDDEN_DIM,
                        batch_first=True
                        )


    def forward(self, x):
        #x.shape:[batch_size,seq_len]
        #计算真实序列长度,张量执行！=操作会逐个元素对比，返回同维度bool变量，普通列表！=操作则是整体比较
        length = (x != self.embed.padding_idx).sum(-1)
        x = self.embed(x)
        #x.shape:[batch_size,seq_len,embed_dim]
        x, _ = self.gru(x)     #得到的x为最后一层所有时间步的隐藏状态

        output=x[range(x.shape[0]),length-1,:]

        return output


class TranslationDecoder(nn.Module):
    def __init__(self,unique_size,padding_index):
        super().__init__()

        self.embed=nn.Embedding(num_embeddings=unique_size,embedding_dim=config.EMBEDDING_DIM,padding_idx=padding_index)
        self.gru=nn.GRU(input_size=config.EMBEDDING_DIM,hidden_size=config.HIDDEN_DIM,batch_first=True)
        self.linear=nn.Linear(config.HIDDEN_DIM,unique_size)

    def forward(self,x,hidden_0):
        #这里解码器是一个token一个token的预测的，并且是自回归生成
        x=self.embed(x)
        output, hidden = self.gru(x, hidden_0)
        output=self.linear(output)
        return output, hidden

class TranslationModel(nn.Module):
    def __init__(self,zh_unique_size,en_unique_size,zh_padding_index,en_padding_index):
        super().__init__()
        self.encoder=TranslationEncoder(zh_unique_size,zh_padding_index)
        self.decoder=TranslationDecoder(en_unique_size,en_padding_index)


