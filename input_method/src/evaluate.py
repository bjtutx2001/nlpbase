import torch

from input_method.src import config
from input_method.src.dataset import get_dataloader
from input_method.src.model import input_method_model


def evaluate(dataloader,device,model):
    model.eval()
    top1_acc_num=0
    top5_acc_num=0
    total=0
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.tolist()[-1]      #取出待预测值
            y_pred = model(x)
            # print(y_pred.shape)
            _,indices=torch.topk(y_pred[:,-1,:], k=5, dim=-1)     #values返回最大值的预测分数，indices返回预测值的索引
            # print(indices.shape)
            for y,y_pred in zip(y,indices.tolist()):
                total+=1
                if y == y_pred[0]:
                    top1_acc_num+=1
                if y in y_pred:
                    top5_acc_num+=1

    return top1_acc_num/total,top5_acc_num/total


def run_evaluate():
    #准备各种资源
    #确定设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #准备测试集数据
    test_dataloader=get_dataloader(train=False)

    #准备词表
    with open(config.MODELS_DIR / 'unique_words.txt', 'r', encoding='utf-8') as f:
        unique_words=f.readlines()
    unique_len=len(unique_words)
    # print(unique_len)
    print(f'词表加载成功')

    #模型
    model = input_method_model(unique_len).to(device)
    model.load_state_dict(torch.load(config.MODELS_DIR / 'best_model.pth'))
    print(f'模型加载成功')

    #进行评估   计算top1准确率和top5准确率
    top1_accu,top5_accu=evaluate(test_dataloader,device,model,)
    print(f'top1准确率：{top1_accu},\ntop5准确率{top5_accu}')




if __name__=='__main__':
    run_evaluate()