#实现自定义神经网络模型
from torch import nn
import config

class input_method_model(nn.Module):
    def __init__(self,unique_size):
        super().__init__()
        #词嵌入层
        #参数1：唯一词表大小，参数2：词向量维度
        self.embedding = nn.Embedding(num_embeddings=unique_size,embedding_dim=config.EMBEDDING_DIM)
        #循环层
        #参数1：词向量维度，参数2：隐藏状态维度,参数3：传入参数批次在前
        self.rnn=nn.RNN(input_size=config.EMBEDDING_DIM,hidden_size=config.HIDDEN_DIM,batch_first=True)
        #输出层
        #参数1：输入维度，这里是隐藏状态维度，参数2：输出维度，这里是唯一词表大小
        self.out = nn.Linear(in_features=config.HIDDEN_DIM,out_features=unique_size)

    def forward(self,x):    #x是输入的序列，size：[batch_size,seq_len],注意此时x还没嵌入
        x=self.embedding(x) #词嵌入后，x：[batch_size,seq_len,embedding_dim]
        output, hidden = self.rnn(x)    #output为最后一层的所有时间步的隐藏状态，hidden为所有层最后一个时间步的隐藏状态
                                        #outputsize:[batch,seq,hidden_dim]
        # output=output[:,-1,:]         #如果是预测一个词，而不是前n个词+后一个词，就需要提出最后一个时间步的隐藏状态

        output=self.out(output)         #最终输出的形状：[batch,seq,unique_size]

        return output