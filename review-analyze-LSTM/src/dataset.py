import pandas as pd
import torch
from torch.utils.data import Dataset,DataLoader
import config


class ReviewAnalyzerDataset(Dataset):
    def __init__(self,path):
        #Dataset父类没有需要初始化的成员
        self.data=pd.read_json(path,lines=True,orient='records').to_dict(orient='records')

    def __len__(self):
        return len(self.data)

    def __getitem__(self,index):
        x=torch.tensor(self.data[index]['review'],dtype=torch.long)
        y=torch.tensor(self.data[index]['label'],dtype=torch.float)
        return x,y

def data_loader(train=True):
    # if train:
    #     path=config.PROCESSED_DIR / 'train.jsonl'
    # else:
    #     path=config.PROCESSED_DIR / 'test.jsonl'
    path=config.PROCESSED_DIR / ("train.jsonl" if train else "test.jsonl")
    dataset=ReviewAnalyzerDataset(path)
    dataloader=DataLoader(dataset,batch_size=config.BATCH_SIZE,shuffle=True)
    return dataloader

if __name__=="__main__":
    data_loader=data_loader(train=True)
    print(len(data_loader))
    for x,y in data_loader:
        print(x,y)
