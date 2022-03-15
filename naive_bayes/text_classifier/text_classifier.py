# 贝叶斯用于文本分类

# 假设存在类别c,计算文档d属于类别c的概率
# calculate the probability of d that is belong to the class of c

# P(c|d) = P(d|c)*P(c)/P(d) 正比于 P（d|c） * P(c) = P(c) *(P(w1|c) * P(w2)|c) * ...* P(w3|c))
# 贝叶斯假设：在计算P（d|c）时,假设文档d的词之间的概念没有关系,即P(d|c) = P(w1|c) * P(w2|c) * ... * P(wn|c)
#           其中d是由w1\w2\w3...\wn组成的
import os
import logging

import jieba


class NaiveBayes:
    def __init__(self, file_dir='.', cls2str={}):
        '''
        Args:
            file_dir:保存的文本数据,文本目录是目录中文本属于的该类别的类名
            class2str:保存的是一个字典,类别名:文本字符串
        '''
        # 记录预料文件的相对路径
        self.file_dir = 'data'
        self.flag = False  # 标识数据是否进行了初始化操作
        self.logger = logging.getLogger()
        self.cls2str = cls2str
        self.cls2word2probability = {}  # 记录每个类别下每个词的概率
        self.clses = []
        self.cls2probability = {}  # 记录每个类别出现的概率

    def process(self):
        # 获取预料文件所在目录绝对路径
        data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.file_dir)
        # 获取所有的类别名
        clses = list(filter(lambda x: x if os.path.isdir(os.path.join(data_path, x)) else False, os.listdir(data_path)))
        self.clses = clses  # 记录文本可能出现的类别 存储字符串列表
        cls2frequency = {}  # 记录每个类别的文本个数 用于计算类别概率
        # 读取各个类别名目录下的所有文件
        for cls in clses:
            str_of_cls = ''  # 记录一个类别的所有文本字符串
            cls_path = os.path.join(data_path, cls)  # 记录某个类别文件的文件夹目录
            cls_files = list(filter(lambda x: x if os.path.isfile(x) else False,
                                    [os.path.join(cls_path, name) for name in os.listdir(cls_path)]))  # 获取某个类比的所有的文件路径
            cls2frequency[cls] = len(cls_files)
            for file in cls_files:
                with open(file, 'r', encoding='utf-8') as f:
                    str_of_cls += f.read() + '\n'
            self.cls2str[cls] = str_of_cls
        # 根据每个类别的文件个数 计算类别的概率
        total_text = sum(map(lambda kv: kv[1], cls2frequency.items()))
        self.cls2probability = {kv[0]: kv[1] / total_text for kv in cls2frequency.items()}
        # 保存各类别样本数据中 每个类别数据中每个词的出现概率
        for cls in clses:
            self.cls2word2probability[cls] = self.cal_probability(self.cls2str[cls])
        self.flag = True

    def infer(self, input_text):
        '''
        根据输入的文本数据计算文本所属类别
        Args:
            input_text: 输入的文本数据

        Returns:
            输出类别字符串
        '''
        cls2probality = {}  # 记录input_text属于每个类别的概率
        if not self.flag:
            self.process()
        words = jieba.lcut(input_text)
        #  P(c) *(P(w1|c) * P(w2)|c) * ...* P(w3|c))
        # P(w|c)
        for cls in self.clses:
            probality = 1
            for word in words:
                probality *= self.cls2word2probability[cls][word]
            cls2probality[cls] = probality
        predict_cls = ('', 0)
        for cls in cls2probality.keys():
            if cls2probality[cls] > predict_cls[1]:
                predict_cls = (cls, cls2probality[cls])
        print(cls2probality)
        return predict_cls

    # 计算每个类别的文本数据中 每个词的出现概率 返回一个字典
    def cal_probability(self, long_str):
        word2frequency = {}
        words = jieba.lcut(long_str)
        count_of_words = 0
        for word in words:
            count_of_words += 1
            if word in word2frequency.keys():
                word2frequency[word] += 1
            else:
                word2frequency[word] = 1
        return {word: word2frequency[word] / count_of_words for word in word2frequency.keys()}


# 可能存在某个词不存在词袋中的情况 此时会报错 可以额外做不存在不存在词时的情况
bayes = NaiveBayes(file_dir='data')
print(bayes.process())
print(bayes.cls2word2probability)
print(bayes.infer('一个垃圾邮件'))
