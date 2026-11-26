import torch
from torch import nn
import config


class ReviewDatasetModel(nn.Module):
    def __init__(self,unique_size,padding_index):
        super().__init__()
        self.embedding=nn.Embedding(unique_size,config.EMBEDDING_DIM,padding_idx=padding_index)
        self.gru=nn.RNN(config.EMBEDDING_DIM,config.HIDDEN_DIM,batch_first=True)
        self.output=nn.Linear(config.HIDDEN_DIM,1)

    def forward(self,x):
        seq_len = (x != self.embedding.padding_idx).sum(dim=-1)  #计算序列长度
        x=self.embedding(x)
        x,hn=self.gru(x)
        batch_index=range(0,x.shape[0])
        last_hidden=x[batch_index,seq_len-1]        #[[],[]]列表索引，前面是行，后面是列，用，隔开，一一对应
        #last_hidden.shape:[batch_size,hidden]，取出每个序列最后一个时间步的隐藏状态，最后一个时间步的索引取决于序列非0时间步长度

        output=self.output(last_hidden).squeeze(-1)
        return output