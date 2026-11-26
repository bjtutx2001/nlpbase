import time
import torch
from datasets import tqdm
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter

from dataset import data_loader
from JiebaTokenizer import JiebaTokenizer
from model import ReviewDatasetModel
from src import config


def train_one_epoch(model, dataloader, criterion, optimzer, device):
    """
    训练一轮
    :param model: 模型
    :param dataloader: 数据
    :param criterion: 损失函数
    :param optimzer: 优化器
    :param device: 设备
    :return: 损失
    """
    model.train().to(device)
    total_loss = 0
    for x, y in tqdm(dataloader,desc="训练进度"):
        x = x.to(device)
        y = y.to(device)
        y_pred = model(x)
        loss = criterion(y_pred, y)
        optimzer.zero_grad()
        loss.backward()
        optimzer.step()
        total_loss += loss.item()
    return total_loss/len(dataloader)


def train():
    #设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #数据
    dataloader=data_loader(train=True)

    #分词器
    tokenizer=JiebaTokenizer.from_words(config.PROCESSED_DIR / "unique_words.txt")

    #模型
    model=ReviewDatasetModel(tokenizer.unique_words_size,tokenizer.pad_token_index)

    #损失函数
    criterion=nn.BCEWithLogitsLoss()

    #优化器
    optimzer=optim.Adam(model.parameters())

    #Tensorboard写入器
    writer=SummaryWriter(log_dir=config.LOGS_DIR / time.strftime("%Y%m%d-%H%M%S"))

    #训练
    for epoch in range(config.EPOCHS):
        print(f'=========={epoch+1}=========')
        loss=train_one_epoch(model,dataloader,criterion,optimzer,device)
        print(f'Loss:{loss:.4f}')
        writer.add_scalar('loss',loss,epoch)

    torch.save(model.state_dict(),config.MODELS_DIR / "best_model.pth")
    print('模型保存成功')

    writer.close()

    print('训练完成')

if __name__=='__main__':
    train()


