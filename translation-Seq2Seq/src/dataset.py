import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset,DataLoader
import config

# 在 PyTorch 中，创建 Dataset 时，不同样本的长度可以不一样。但是在 DataLoader 将多个样本组装成一个批次时，默认情况下要求对应的张量形状一致。
# DataLoader 在获取单个样本时并不要求样本长度一致，而是在将多个样本组装成一个 Batch 时，默认要求对应的张量形状一致。
class TranslationDataset(Dataset):
    def __init__(self,path):
        #Dataset父类没有需要初始化的成员
        self.data=pd.read_json(path,lines=True,orient='records').to_dict(orient='records')

    def __len__(self):
        return len(self.data)

    def __getitem__(self,index):
        x=torch.tensor(self.data[index]['zh'],dtype=torch.long)
        y=torch.tensor(self.data[index]['en'],dtype=torch.long)
        return x,y

def collate_fn(batch):
    #batch是一批数据，有输入和预测组成的二元组：[(x1,y1),(x2,y2)。。。。。。]
    x=[item[0] for item in batch]
    y=[item[1] for item in batch]

    x=pad_sequence(x,batch_first=True,padding_value=0,padding_side='right')
    y=pad_sequence(y,batch_first=True,padding_value=0)          #填充方向默认右边
    return x,y



def data_loader(train=True):
    # if train:
    #     path=config.PROCESSED_DIR / 'train.jsonl'
    # else:
    #     path=config.PROCESSED_DIR / 'test.jsonl'
    path=config.PROCESSED_DIR / ("train.jsonl" if train else "test.jsonl")
    dataset=TranslationDataset(path)
    dataloader=DataLoader(dataset,batch_size=config.BATCH_SIZE,shuffle=True,collate_fn=collate_fn)
    #collate_fn是PyTorch中DataLoader的一个参数，用于指定如何将Dataset中取出的多个样本组装成一个Batch。   传入的是函数对象
    return dataloader

if __name__=="__main__":
    data_loader=data_loader(train=True)
    print(len(data_loader))
    for x,y in data_loader:
        print(x,y)
