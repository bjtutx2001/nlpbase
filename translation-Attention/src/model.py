from torch import nn
import torch
from src import config

class Attention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,decoder_hidden,encoder_outputs):
        encoder_outputs=encoder_outputs.transpose(1,2)
        Attention_score=torch.bmm(decoder_hidden,encoder_outputs)
        Attention_weights=torch.softmax(Attention_score,dim=-1)

        return torch.bmm(Attention_weights,encoder_outputs.transpose(1,2))



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
        hidden_states, _ = self.gru(x)     #得到的x为最后一层所有时间步的隐藏状态

        last_hidden_state=hidden_states[range(x.shape[0]),length-1,:]

        return hidden_states,last_hidden_state


class TranslationDecoder(nn.Module):
    def __init__(self,unique_size,padding_index):
        super().__init__()

        self.embed=nn.Embedding(num_embeddings=unique_size,embedding_dim=config.EMBEDDING_DIM,padding_idx=padding_index)
        self.gru=nn.GRU(input_size=config.EMBEDDING_DIM,hidden_size=config.HIDDEN_DIM,batch_first=True)
        self.linear=nn.Linear(config.HIDDEN_DIM * 2,unique_size)
        self.attention=Attention()

    def forward(self,x,last_hidden_state,hidden_states):
        #这里解码器是一个token一个token的预测的，并且是自回归生成
        x=self.embed(x)
        output, hidden = self.gru(x, last_hidden_state)

        #应用注意力机制（解码器当前时间步的隐藏状态与编码器所有时间步的隐藏状态做注意力运算）
        context_vector=self.attention(output,hidden_states)
        output=torch.cat((output,context_vector),dim=-1)

        output=self.linear(output)
        return output, hidden

class TranslationModel(nn.Module):
    def __init__(self,zh_unique_size,en_unique_size,zh_padding_index,en_padding_index):
        super().__init__()
        self.encoder=TranslationEncoder(zh_unique_size,zh_padding_index)
        self.decoder=TranslationDecoder(en_unique_size,en_padding_index)


