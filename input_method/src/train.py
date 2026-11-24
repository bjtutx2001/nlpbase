#定义训练函数
import time
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch import nn
import torch.optim as optim
import config
from input_method.src.dataset import get_dataloader
from input_method.src.model import input_method_model


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    训练一个轮次
    :param model: 模型
    :param dataloader: 数据
    :param criterion: 损失函数
    :param optimizer: 优化器
    :param device: 设备
    :return: 当前轮次平均loss
    """
    total_loss=0
    for x, y in tqdm(dataloader,desc='本轮训练进度'):
        x = x.to(device)
        y = y.to(device)
        #前向传播
        y_pred = model(x)
        loss = criterion(y_pred.permute(0,2,1), y)
        #反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss+=loss.item()

    return total_loss / len(dataloader)   #返回平均损失


def train():
    #1、确定训练设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #2、加载数据集
    dataloader=get_dataloader(train=True)

    #加载唯一词表
    with open(config.MODELS_DIR / 'unique_words.txt','r',encoding='utf-8') as f:
        # unique_words = f.read().splitlines()  #按照将字符串拼接成列表
        # unique_words = f.readlines()    #按行读取，每行+换行符，返回列表
        # unique_words = f.read()             #返回字符串
        unique_words = [i.strip() for i in f.readlines()]
        # print(len(unique_words))
        # print(unique_words)
        unique_size=len(unique_words)

    #4、初始化模型
    model=input_method_model(unique_size).to(device)
    model.train()
    #5、定义损失函数
    criterion = nn.CrossEntropyLoss()

    #6、定义优化器
    optimizer=optim.Adam(model.parameters(),lr=config.LEARNING_RATE)

    #7、定义写入器，实时记录训练过程中的损失变化，需要用到TensorBoard（画图）和SummaryWriter（记录数据）
    writer=SummaryWriter(log_dir=config.LOGS_DIR / time.strftime("%Y%m%d-%H%M%S"))

    #定义最佳损失，当损失小于这个值时就保存模型
    best_loss=float('inf')  #inf是infinity无限的缩写

    #8、开始训练
    for epoch in range(config.EPOCHS):
        #传统写法
        # for x, y in dataloader:
        #     x=x.to(device)
        #     y=y.to(device)
        #     y_pred=model(x)
        #     loss=criterion(y_pred,y)
        #     #梯度清零+反向传播+更新参数
        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()

        print('='*10,f'轮次：{epoch+1}','='*10)
        loss=train_one_epoch(model,dataloader,criterion,optimizer,device)
        print(f'loss:{loss}')

        #记录训练结果
        #格式：标签/图标名称，y值，x值
        writer.add_scalar('loss',loss,epoch)

        # #保存模型
        # if loss<best_loss:
        #     best_loss=loss
        #     torch.save(model.state_dict(),config.MODELS_DIR / 'best_model.pth')
        #     print('保存成功！')
    torch.save(model.state_dict(), config.MODELS_DIR / 'best_model.pth')
    print('模型保存成功！')

    #关闭写入器
    writer.close()
    print('训练完成！')


if __name__=="__main__":
    train()