import torch
from torch import nn

from JiebaTokenizer import JiebaTokenizer
from src import config
from model import ReviewDatasetModel


def predict(device, model, tokenizer, sentences):
    index=tokenizer.encode(sentences,config.SEQ_LEN)
    tensor_index=torch.tensor([index],dtype=torch.long).to(device)

    with torch.no_grad():
        y_pred = model(tensor_index)
        y_pred = torch.sigmoid(y_pred)
        y_pred = y_pred.tolist()

    return y_pred[0]
def run_predict():
    #选择各种资源
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = JiebaTokenizer.from_words(config.PROCESSED_DIR / "unique_words.txt")
    model = ReviewDatasetModel(tokenizer.unique_words_size,tokenizer.pad_token_index)
    model.load_state_dict(torch.load(config.MODELS_DIR / "best_model.pth"))
    model.eval()
    model.to(device)

    #输入评价
    print('请输入评价(输入q或quit退出)：')
    while True:
        user_input=input('>')   #输入提示
        if user_input in ['q','quit']:
            break
        if user_input.strip()=='':
            print('请输入内容')
            continue
        else:
            result=predict(device,model,tokenizer,user_input)
            if result >= 0.5:
                print('正向')
            else:
                print('负向')

if __name__=='__main__':
    run_predict()