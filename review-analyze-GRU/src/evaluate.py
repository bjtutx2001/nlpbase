import torch

import config
from dataset import data_loader
from input_method.src import tokenizer
from model import ReviewDatasetModel
from JiebaTokenizer import JiebaTokenizer


def evaluate(dataloader,device,model):
    model.eval()
    total_count=0
    correct_count=0
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.tolist()
            y_pred = model(x)
            y_pred=torch.sigmoid(y_pred)
            y_pred=y_pred.tolist()
            for y_pred,y in zip(y_pred,y):
                y_pred=1 if y_pred >= 0.5 else 0
                total_count += 1
                if y==y_pred:
                    correct_count+=1
    return correct_count/total_count


def run_evaluate():
    #准备各种资源
    #确定设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #准备测试集数据
    test_dataloader=data_loader(train=False)

    #准备词表
    tokenizer=JiebaTokenizer.from_words(config.PROCESSED_DIR / "unique_words.txt")

    #模型
    model = ReviewDatasetModel(tokenizer.unique_words_size,tokenizer.pad_token_index).to(device)
    model.load_state_dict(torch.load(config.MODELS_DIR / 'best_model.pth'))
    print(f'模型加载成功')

    #进行评估
    acc=evaluate(test_dataloader,device,model)
    print(f'准确率：{acc}')




if __name__=='__main__':
    run_evaluate()