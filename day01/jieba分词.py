"""
分词过程就是找到分界符的过程，每个分词结果就是一个token
常见分词包：
    jieba：精确模式，全模式，搜索引擎模式，中文繁体，用户自定义词典
    IK分词器：elasticsearch搜索引擎用的多
    SnowNLP
    pyltp
    THULAC
    jieba的分词模式，其实就是分词粒度不同，例如“软件工程”——>软件工程（粗粒度），软件，工程（细粒度）
    jieba属于第三方库，用之前需要先安装 pip install jieba
"""
from importlib.resources import contents

import jieba
#todo 1.定义函数，演示jieba精确分词模式，适用于：文本分析
"""
精确模式会自动选择覆盖字符最长，最完整的词，概率最大的分词结果
例：自定义了通信 工程两个词，但是jieba自带词库含有通信工程这四个字组成的字库，他会优先分词为通信工程
而非分词为通信和工程。
"""
# print(__name__)
def dm01():
    #1.定义待分词内容
    content = "我在北京交通大学读研究生，目前研二，专业是通信工程，研究方向是无线通信"
    #print(content)

    #2.使用jieba分词进行精确模式分词（默认模式）——>cut_all = False
    result1 = jieba.cut(content, cut_all=False)
    print(f"result1: {result1}") #generator object Tokenizer.cut at 0x000002173BB3F920
    #result1是一个生成器对象，只有要用的时候才读数据，好处：省内存，只遍历一次

    #3.从生成器中，提取所有分词结果
    #3.1思路1：next()函数，逐个获取下个元素
    print(next(result1))
    print(next(result1))
    print("-"*20)

    #3.2思路2遍历方式，从生成器提取元素
    for item in result1:
        print(item)
    print("-" * 20)

    #4.如果要列表怎么办
    #4.1思路1，直接把上述生成器转成列表
    list1 = list(result1)
    print(f"list1: {list1}")

    #4.2思路2，切词时返回list，用lcut方法，相对于：语法糖
    list2 = jieba.lcut(content, cut_all=False)
    print(f"list2: {list2}")


#todo 2.定义函数，演示jieba全模式分词模式（能拆的全拆），适用于：关键词提取，不需要严格分词准确性的场景
# print(__name__)
def dm02():
    #1.定义待分词内容
    content = "我在北京交通大学读研究生，目前研二，专业是通信工程，研究方向是无线通信"
    #print(content)

    #2.使用jieba分词进行全模式分词——>cut_all = True
    result1 = jieba.cut(content, cut_all=True)
    print(f"result1: {result1}") #generator object Tokenizer.cut at 0x000002173BB3F920
    #result1是一个生成器对象，只有要用的时候才读数据，好处：省内存，只遍历一次

    #3.从生成器中，提取所有分词结果
    #3.1思路1：next()函数，逐个获取下个元素
    print(next(result1))
    print(next(result1))
    print("-"*20)

    #3.2思路2遍历方式，从生成器提取元素
    for item in result1:
        print(item)
    print("-" * 20)

    #4.如果要列表怎么办
    #4.1思路1，直接把上述生成器转成列表
    list1 = list(result1)
    print(f"list1: {list1}")

    #4.2思路2，切词时返回list，用lcut方法，相对于：语法糖
    list2 = jieba.lcut(content, cut_all=True)
    print(f"list2: {list2}")

#todo 3.定义函数，演示jieba搜索引擎模式，适用于：搜索引擎分词，文本匹配
"""
模式解释：
    搜索引擎模式就是在精确模式基础上，对长词进行再次切分，提高召回率
例如：
    场景1：用户输入：程序员
    精确模式：只能匹配包含完整"程序员"的文档
    搜索引擎模式：不仅能匹配"程序员"文档，还能搜索"程序"和"员"的文档，提高召回率
    
    场景2：实际应用场景（电商搜索），商品标题为："苹果手机保护套"，用户搜索"苹果套"
    精确模式：   无法匹配，分词为（“苹果”，“手机”，“保护套”）
    搜索引擎模式：能匹配，分词为（“苹果”，“手机”，“保护”，“套”）
"""
def dm03():
    #1.定义待分词内容
    content = "我在北京交通大学读研究生，目前研二，专业是通信工程，研究方向是无线通信"
    #print(content)

    #2.使用jieba分词进行搜索引擎模式分词——>cut_for_search()
    result1 = jieba.cut_for_search(content)
    print(f"result1: {result1}") #generator object Tokenizer.cut at 0x000002173BB3F920
    #result1是一个生成器对象，只有要用的时候才读数据，好处：省内存，只遍历一次

    #3.从生成器中，提取所有分词结果
    #3.1思路1：next()函数，逐个获取下个元素
    print(next(result1))
    print(next(result1))
    print("-"*20)

    #3.2思路2遍历方式，从生成器提取元素
    for item in result1:
        print(item)
    print("-" * 20)

    #4.如果要列表怎么办
    #4.1思路1，直接把上述生成器转成列表
    list1 = list(result1)
    print(f"list1: {list1}")

    #4.2思路2，切词时返回list，用lcut方法，相对于：语法糖
    list2 = jieba.lcut_for_search(content)
    print(f"list2(搜索引擎模式分词): {list2}")

    #5。拓展，打印三种模式的分词结果，对比一下
    list3 = jieba.lcut(content, cut_all=False)
    print(f"list3(精确模式): {list3}")
    list4 = jieba.lcut(content, cut_all=True)
    print(f"list4(全模式): {list4}")
    list5 = jieba.lcut_for_search(content)
    print(f"list5(搜索引擎模式): {list5}")

#todo 4.定义函数，演示jieba繁体字分词
def dm04():
    #1。定义分词句子
    content = "春風得意馬蹄疾，一日看盡長安花"

    #2.使用精确模式进行繁体字分词
    result1 = jieba.lcut(content, cut_all=False) #精确模式

    #3.打印
    print(f"result1(精确模式分词繁体字): {result1}")


#todo 5.定义函数，演示jieba的自定义词典功能
"""
添加自定义词典后，jieba能准确识别词典中出现的词汇，提升整体的识别准确率
词典（文件的）格式，每一行分三个部分，分别是
    词语（必有） 词频（可选） 词性（可选） 中间用空格隔开，顺序不能颠倒
词频大小的功能是如果出现类似的词时，分词结果优先取词频大的词汇,词性的具体写法，参考结巴自定义词汇词性表
"""
def dm05():
    #1.定义待分词的文本内容
    content = "我在北京交通大学读研究生，目前研二，专业是通信工程，研究方向是无线通信"

    #2.执行未加载自定义词典的分词结果
    list1 = jieba.lcut(content) #精确模式分词
    print(f"list1(精确模式)：{list1}")

    #3.加载用户自定义词典
    jieba.load_userdict("./Data/dictionary.txt")

    #4.执行加载自定义词典分词结果
    list2 = jieba.lcut(content)  # 精确模式分词
    print(f"list2(加载词典精确模式)：{list2}")



#todo 6.测试代码
if __name__ == "__main__":
    #测试精确分词
    # dm01()
    #测试全模式
    # dm02()
    #测试搜索引擎模式
    # dm03()
    #繁体字分词
    # dm04()
    dm05()