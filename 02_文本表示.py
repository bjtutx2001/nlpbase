"""
文本表示是自然语言处理（Natural Language Processing，NLP）中的基础环节，
指的是将人类能够理解的自然语言文本转换为计算机能够处理的数值形式（向量、矩阵或张量）的过程。
传统方法：
    1、one-hot编码
    2、bag of words词袋编码
    3、N-gram
    4、TF-IDF
现代文本表示：
    1、word2Vec
    2、Fasttext
    3、Embedding
    4、BERT
英文分词：按照分词粒度可以分为词级分词（按照完整的单词切词），字符级分词（字母，符号，空格为最小分词单位）和子词分词（按照词根词缀分词）
            缺点：          OOV问题，出现不再词表的词       语义信息极低，训练成本极高               最接近人类的分词，目前最实用
BPE算法（byte pair encoding）：最早被广泛用于子词分词的算法，详细实现参考Hugging Face内的教程

jieba：中文分词常用分词器，开源
"""