import jieba
import pandas as pd
import torch
import config
from input_method.src.model import input_method_model


def predict(text,model,word2index,device,index2word):
    # #选择设备
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # #加载唯一词
    # with open(config.MODELS_DIR / 'unique_words.txt', 'r',encoding='utf-8') as f:
    #     unique_words = f.read().splitlines()
    # # print(unique_words)
    # unique_size = len(unique_words)
    # # print(word)
    # #加载词表
    # word2index=pd.read_json(config.MODELS_DIR / 'word2index.json',typ='series').to_dict()
    # index2word={index:word for word,index in word2index.items()}
    # # print(word2index)
    # # unique_size_2=len(word2index)
    # # print(unique_size == unique_size_2)
    # # print(index)

    # #加载模型
    # model=input_method_model(unique_size).to(device)
    # model.load_state_dict(torch.load(config.MODELS_DIR / 'best_model.pth',map_location=device))
    # model.eval()

    #处理输入
    words=jieba.lcut(text)
    index=[word2index.get(word,0) for word in words]
    index=torch.tensor([index],dtype=torch.long).to(device)

    #预测
    with torch.no_grad():   #预测时关闭梯度计算，减少内存
        y_pred=model(index)
        print(y_pred.shape)
        #参数1：筛选的张量，参数2：选取的元素数量，参数3：沿着哪个维度筛选,参数4：最大值，默认True，参数5：排序，默认True
        values, indices = torch.topk(y_pred, k=5, dim=-1,largest=True,sorted=True)
        print(values,f'\n',indices)
    #转成数组
    pred_index=indices.cpu().squeeze(0).tolist()
    # print(pred_index)
    pred_word=[]
    #预测所有补偿
    # for words_index in pred_index:
    #     temp=[]
    #     for word_index in words_index:
    #         temp.append(index2word[word_index])
    #     pred_word.append(temp)

    #预测最后一个步长,      注意，python原生列表支持在同一个[]中使用逗号风格多个维度进行索引
    for word in pred_index[-1]:
        pred_word.append(index2word[word])
    print(pred_word)

def run_predict():
    #准备各种资源
    # 选择设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载唯一词
    with open(config.MODELS_DIR / 'unique_words.txt', 'r', encoding='utf-8') as f:
        unique_words = f.read().splitlines()
    # print(unique_words)
    unique_size = len(unique_words)
    # print(word)
    # 加载词表
    word2index = pd.read_json(config.MODELS_DIR / 'word2index.json', typ='series').to_dict()
    index2word = {index: word for word, index in word2index.items()}
    # print(word2index)
    # unique_size_2=len(word2index)
    # print(unique_size == unique_size_2)
    # print(index)

    # 加载模型
    model = input_method_model(unique_size).to(device)
    model.load_state_dict(torch.load(config.MODELS_DIR / 'best_model.pth', map_location=device))
    model.eval()

    #用户输入
    print('欢迎使用输入法模型(输入q或者quit退出)')
    input_list=[]
    while True:
        user_input=input('>')  #提示用户输入
        if user_input in ['q','quit']:
            break
        elif user_input.strip()=='':
            print('请输入内容')
            continue
        else:
            input_list.append(user_input)
            sentence=''.join(input_list)
            print(f'输入历史：{sentence}')
            predict(sentence,model,word2index,device,index2word)


if __name__ == '__main__':
    # predict('我们团队')
    run_predict()