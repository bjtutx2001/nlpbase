import torch
from torch import nn

from JiebaTokenizer import EnglishTokenizer,ChineseTokenizer
from src import config
from model import TranslationModel


def predict_batch(model,input_tensor,device,entokenizer):
    """
    预测一批输入
    :param model: 模型
    :param input_tensor: 一批的张量
    :param device: 设备
    :return: 一批的预测结果
    """
    with torch.no_grad():

        #编码
        context_vector=model.encoder(input_tensor)  #编码结果

        #解码
        hidden_h=context_vector.unsqueeze(0)       #初始隐藏状态，要增加一个维度
        batch_size=context_vector.size(0)
        decoder_input=torch.full([batch_size,1],entokenizer.sos_token_index,dtype=torch.long,device=device)   #初始输入,全sos
        generate_index=[]
        is_ceased=torch.full([batch_size],False,device=device)
        #自回归生成
        for i in range(config.SEQ_MAX):
            #生成
            decoder_output,hidden_output=model.decoder(decoder_input,hidden_h)       #[batch_size,1,unique_size]
            #保存结果
            decoder_output=torch.argmax(decoder_output,dim=-1)
            generate_index.append(decoder_output)       #[tensor[所有批次第一步长输出]，tensor[所有批次第二步长输出]。。。]
            #更新输入和隐藏状态
            decoder_input=decoder_output
            hidden_h=hidden_output
            #判断是否应该结束
            is_ceased |= (entokenizer.eos_token_index == decoder_output.squeeze(1))
            if is_ceased.all():
                break
        generate_index=torch.cat(generate_index,dim=1)     #[batch_size,seq]
        generate_index=generate_index.tolist()
        for index,sentence_index in enumerate(generate_index):
            if entokenizer.eos_token_index in sentence_index:
                end_index=sentence_index.index(entokenizer.eos_token_index)     #获取指定内容第一次出现的索引
                generate_index[index]=sentence_index[:end_index]
    return generate_index




def predict(device,model,zh_tokenizer,en_tokenizer, sentences):
    #处理输入
    indexes=zh_tokenizer.encode(sentences)
    input_tensor=torch.tensor([indexes],dtype=torch.long)   #输入要是二维的
    input_tensor=input_tensor.to(device)

    #预测逻辑
    bacth_result=predict_batch(model,input_tensor,device,en_tokenizer)


    return en_tokenizer.decode(bacth_result[0])

def run_predict():
    #选择各种资源
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    en_tokenizer = EnglishTokenizer.from_words(config.PROCESSED_DIR / "en_unique_words.txt")
    zh_tokenizer = ChineseTokenizer.from_words(config.PROCESSED_DIR / "zh_unique_words.txt")
    model = TranslationModel(zh_tokenizer.unique_words_size,en_tokenizer.unique_words_size,zh_tokenizer.pad_token_index,en_tokenizer.pad_token_index)
    model.load_state_dict(torch.load(config.MODELS_DIR / "best_model.pth",map_location=device))
    model.eval()
    model.to(device)
    #输入评价
    print('欢迎使用中英翻译模型(输入q或quit退出)：')
    while True:
        user_input=input('>')   #输入提示
        if user_input in ['q','quit']:
            break
        if user_input.strip()=='':
            print('请输入中文')
            continue
        else:
            result=predict(device,model,zh_tokenizer,en_tokenizer,user_input)
        print(result)

if __name__=='__main__':
    run_predict()