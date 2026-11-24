from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
import config


#实现自定义Dataset
class my_dataset(Dataset):
    def __init__(self,path):
        self.data=pd.read_json(path,orient='records',lines=True).to_dict(orient='records')          #转成字典列表

    #调用len(obj)会实现下述魔法方法
    def __len__(self):
        return len(self.data)
    #对对象进行索引操作，obj[]时，会实现下述魔法方法
    def __getitem__(self,index):
        input=torch.tensor(self.data[index]['x'],dtype=torch.long)
        target=torch.tensor(self.data[index]['y'],dtype=torch.long)
        return input,target

def get_dataloader(train=True):
    path=config.PROCESSED_DATA_DIR / ('train_dataset.jsonl' if train else 'test_dataset.jsonl')
    dataset=my_dataset(path)
    dataloader=DataLoader(dataset,batch_size=config.BATCH_SIZE,shuffle=True,num_workers=2)
    return dataloader

if __name__=='__main__':
    dataloader=get_dataloader(train=False)
    print(len(dataloader))      #得到的是数据的总批次
    for x,y in dataloader:
        print(x.shape)          #从加载器中取出输入和输出，输入和输出的纬度为：[batch_size,seq_len]
        print(y.shape)          #因为数据加载器是按每批多少个读取数据的
