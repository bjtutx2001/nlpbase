"""
演示构建RNN
"""
import torch
import torch.nn as nn

#创建RNN：输入词向量维度：4，输出隐藏状态维度：6，rnn层数:2,启用双向rnn，批次维度在前
rnn=nn.RNN(input_size=4,hidden_size=6,num_layers=2,bidirectional=True,batch_first=True)

# inputsize：[batch_size,seq_len,input_size]
input=torch.randn(size=(3,5,4))

#outputsize:[batch_size,seq_len,2*hidden_size]
#hnsize:[num_layers*num_directions,batch_size,hidden_size]
output=rnn.forward(input)
for name,param in rnn.named_parameters():
    print(f'{name}:{param.shape}')
