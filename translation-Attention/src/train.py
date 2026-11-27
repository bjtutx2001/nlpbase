import time
import torch
from datasets import tqdm
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter

from dataset import data_loader
from JiebaTokenizer import BaseTokenizer,EnglishTokenizer,ChineseTokenizer
from model import TranslationDecoder,TranslationEncoder,TranslationModel
from src import config


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    训练一轮
    :param model: 模型
    :param dataloader: 数据
    :param criterion: 损失函数
    :param optimzer: 优化器
    :param device: 设备
    :return: 损失
    """
    model.train()
    total_loss = 0
    for x, y in tqdm(dataloader,desc="训练进度"):
        x, y = x.to(device), y.to(device)

        #前向传播
        #编码阶段
        hidden_states,hidden_h=model.encoder(x)      #编码器得到最后一层最后一个时间步的隐藏状态，包含所有时间步的信息，作为解码器的输入
        hidden_h=hidden_h.unsqueeze(0)    #编码器得到的h0维度为[batch_size,hidden_size],作为初始隐藏状态输入解码要扩展维度

        #解码阶段
        decoder_inputs = y[:,:-1]        #前n-1个词作为解码器输入
        decoder_targets = y[:,1:]         #后n-1个词作为目标
        decoder_outputs=[]
        #Teacher Forcing（教师强制），一个一个输入，一输入次数做循环
        seq_len=decoder_inputs.shape[1]
        for i in range(seq_len):
            decoder_input=decoder_inputs[:,i].unsqueeze(1)   #取一个词出来后是一维的，要求输入是二维的
            decoder_output, hidden_h=model.decoder(decoder_input,hidden_h,hidden_states)
            decoder_outputs.append(decoder_output)

        #计算损失
        decoder_outputs=torch.cat(decoder_outputs,dim=1)
        decoder_outputs=decoder_outputs.permute(0,2,1)
        Loss=criterion(decoder_outputs,decoder_targets)

        #反向传播
        optimizer.zero_grad()
        Loss.backward()
        optimizer.step()
        total_loss += Loss.item()


    return total_loss


def train():
    #设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #数据
    dataloader=data_loader(train=True)

    #分词器
    zh_tokenizer=ChineseTokenizer.from_words(config.PROCESSED_DIR / "zh_unique_words.txt")
    en_tokenizer=EnglishTokenizer.from_words(config.PROCESSED_DIR / "en_unique_words.txt")

    #模型
    model=TranslationModel(zh_tokenizer.unique_words_size,en_tokenizer.unique_words_size,zh_tokenizer.pad_token_index,
                           en_tokenizer.pad_token_index).to(device)

    #损失函数
    criterion=nn.CrossEntropyLoss(ignore_index=en_tokenizer.pad_token_index)   #指定不参与损失计算的目标索引

    #优化器
    optimizer=optim.Adam(model.parameters(),lr=config.LEARNING_RATE)

    #Tensorboard写入器
    writer=SummaryWriter(log_dir=config.LOGS_DIR / time.strftime("%Y%m%d-%H%M%S"))

    best_loss=float('inf')
    #训练
    for epoch in range(config.EPOCHS):
        print(f'=========={epoch+1}=========')
        loss=train_one_epoch(model,dataloader,criterion,optimizer,device)
        print(f'Loss:{loss:.4f}')
        writer.add_scalar('loss',loss,epoch)
        if loss<best_loss:
            best_loss=loss
            torch.save(model.state_dict(),config.MODELS_DIR / "best_model.pth")


    print('模型保存成功')

    writer.close()

    print('训练完成')

if __name__=='__main__':
    train()

