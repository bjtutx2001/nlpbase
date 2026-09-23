"""
案例：
    代码演示下文本预处理之词性标注（part-of-speech tagging POS）
    例如：名词、动词、形容词
名词解释：
    NER：Named Entity Recognition，命名实体识别，就是识别出一段文本中可能存在的命名实体
    作用：和词汇（分词）一样，也是人类理解文本的基本单元
"""
#导包
import jieba.posseg as pseg

#1.定义变量，记录：待分词并标注词性的文本内容
content = "我爱北京天安门和北京交通大学"

#2.使用jieba进行分词和词性标注
result = pseg.lcut(content)
print(f"result(分词与词性标注结果)：{result}")

#3.获取每组数据的词和词性
for word,flag in result:
    print(f"词语:{word:>6}，\t词性:{flag}")