import torch
from nltk.translate.bleu_score import corpus_bleu

import config
from dataset import data_loader
from input_method.src import tokenizer
from model import TranslationModel
from JiebaTokenizer import EnglishTokenizer,ChineseTokenizer
from src.predict import predict, predict_batch


def evaluate(dataloader,device,model,en_tokenizer):
    predictions=[]      #[[],[],[]]
    references=[]       #[[[],[]],[[],[],[]],[[],[],[]]]

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device).tolist()

        batch_result=predict_batch(model,x,device,en_tokenizer)     #[[],[],[]]
        predictions.extend(batch_result)
        references.extend([[target[1:target.index(en_tokenizer.eos_token_index)]] for target in y])


    return corpus_bleu(references,predictions)


def run_evaluate():
    #准备各种资源
    #确定设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #准备测试集数据
    test_dataloader=data_loader(train=False)

    #准备词表
    en_tokenizer=EnglishTokenizer.from_words(config.PROCESSED_DIR / "en_unique_words.txt")
    zh_tokenizer=ChineseTokenizer.from_words(config.PROCESSED_DIR / "zh_unique_words.txt")

    #模型
    model = TranslationModel(zh_tokenizer.unique_words_size,en_tokenizer.unique_words_size,zh_tokenizer.pad_token_index,en_tokenizer.pad_token_index).to(device)
    model.load_state_dict(torch.load(config.MODELS_DIR / 'best_model.pth',map_location=device))
    model.eval()
    print(f'模型加载成功')

    #进行评估
    acc=evaluate(test_dataloader,device,model,en_tokenizer)
    print(f'bleu：{acc}')




if __name__=='__main__':
    run_evaluate()