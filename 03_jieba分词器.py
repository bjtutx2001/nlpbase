"""
案例：演示jieba分词器
"""
import jieba

text='我在北京交通大学读通信工程专业，今年研三'

#cut返回生成器
l1=jieba.cut(text,cut_all=True)    #cut_all表示全分词模式
l2=jieba.cut(text)
#lcut返回列表，存储切词结果
l3=jieba.lcut(text)
l4=jieba.lcut_for_search(text)
#查看结果
print(f'{l1}\n{l2}\n{l3}\n{l4}')

#jieba支持自定义词典