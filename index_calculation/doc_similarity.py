import math
from functools import reduce

"""
bm25的计算原理：
1)场景:通常用于计算文本之间的相似度;
2)问题:假设存在样本Q和目标d两个文本,需要计算Q与d的相似度;
3)实现步骤:
    1.1 分词:
        对Q进行分词 Q = [q1,q2,q3,...qi...,qn]
    1.2 qi与d相关性R(qi,d)得分计算:
        R(qi,d) = (fi ( k1 + 1 ) / ( fi + K )) * ( qfi ( k2 + 1 ) / qfi + k2 ),
        K = k1 * ( 1 - b + b * dl / avgdl)
        其中,k1、k2、b为调节参数,按经验一般k1=2,b=0.75
        fi:qi在d中的频率
        qfi:qi在Q中的频率
        dl:d的长度
        avgdl:所有文档的平均长度
    1.3 R(qi,d)的权重wi得分计算:
        wi表示的是一个词语与d的相关性权重
        可以采用IDF计算:IDF = log( 语料库中文档总数 / ( 包词qi的文档数 + 1 ))
    1.4 样本Q与目标d相似度计算:
        1<=i<=n,sum(qi * R(qi,d)) 
"""


class Docs:
    def __init__(self, docs):
        """
        Args:
            docs:doc的列表 列表的每一个元素是一个doc分词列表
        """
        self.flag = False  # 标识Docs对象是否执行过init方法 进行必要信息的初始化
        self.docs = docs
        self.num_docs = None
        self.doc_avg_len = None
        self.word2idf = {}  # 存储每个词在预料中的逆文档频率 idf = log(文档个数/(包含指定词的文档个数 + 1))

    def init(self):
        if not self.flag:
            self.num_docs = len(self.docs)
            docs_length = [len(doc) for doc in self.docs]
            self.doc_avg_len = reduce(lambda x, y: x + y, docs_length) / self.num_docs
            # 统计每个词对应的doc个数
            for doc in self.docs:
                # 针对一个doc中的词 每个词只统计一次
                for word in set(doc):
                    if word in self.word2idf.keys():
                        self.word2idf[word] += 1
                    else:
                        self.word2idf[word] = 1
            # 计算每个词的IDF
            self.word2idf = {math.log(self.num_docs / (self.word2idf[word] + 1)) for word in self.word2idf.keys()}
            # 添加预料库中不存在词的IDF
            self.word2idf['INEXISTENCE'] = math.log(self.num_docs)

    def cal_bm25(self, input_doc_tokens, target_doc_tokens, k1=2, k2=2, b=0.75):
        dword_freq_in_d = {}  # d中的词的词频
        for word in target_doc_tokens:
            if word in dword_freq_in_d.keys():
                dword_freq_in_d[word] += 1
            else:
                dword_freq_in_d[word] = 1
        # Q中的词频统计
        qword_freq_in_Q = {}  # q中的词的词频
        for word in input_doc_tokens:
            if word in qword_freq_in_Q.keys():
                qword_freq_in_Q[word] += 1
            else:
                qword_freq_in_Q[word] = 1
        # d的长度
        d_length = len(target_doc_tokens)
        # 计算q中的词在d中的频率
        qword_freq_in_d = {word: dword_freq_in_d[word] if word in dword_freq_in_d.keys() else 0 for word in
                           qword_freq_in_Q.keys()}
        word2ri = {}
        for word in qword_freq_in_Q.keys():
            K = k1 * (1 - b + b * d_length / self.doc_avg_len)
            ri = qword_freq_in_d[word] * (k1 + 1) / (qword_freq_in_d[word] + K) * (
                    qword_freq_in_Q[word] * (k2 + 1) / (qword_freq_in_Q[word] + k2))
            word2ri[word] = ri
        # 计算wi:IDF = log( 语料库中文档总数 / ( 包词qi的文档数 + 1 ))
        word2wi = [self.word2idf[word] if word in self.word2idf.keys() else self.word2idf['INEXISTENCE'] for word in
                   qword_freq_in_Q.keys()]
        similarity = 0
        for word in qword_freq_in_Q.keys():
            similarity += word2wi[word] * word2ri[word]
        return similarity

    # 计算input_doc_tokens中每个词的词频
    def cal_tf(self, input_doc_tokens):
        if not input_doc_tokens:
            return {}
        word2tf = {}
        for word in input_doc_tokens:
            if word in word2tf.keys():
                word2tf[word] += 1
            else:
                word2tf[word] = 1
        # 考虑到文章有长短之分,为了便于不同长度文章之间的比较 词频进行"标准化"
        return {word: word2tf[word] / len(input_doc_tokens) for word in word2tf.keys()}

    def cal_tf_idf(self, input_doc_tokens):
        tf = self.cal_tf(input_doc_tokens)
        idf = [self.word2idf[word] for word in input_doc_tokens]
        word2tfidf = {}
        for word in set(input_doc_tokens):
            word2tfidf[word] = tf[word] * idf[word]
        return word2tfidf

    # 场景：用于比较一个句子与另一个句子的相似度 相似度越大越相似
    def get_similarity_by_tf_idf(self, input_doc_tokens, targets_doc_tokens, top_n):
        # 计算输入语句中每个词的tf-idf
        tf_idfs = self.cal_tf_idf(input_doc_tokens)
        # 按tf-idf排序 取前n个word
        # 降序 取出top_n个关键字
        keywords = map(lambda kv: kv[0], sorted(tf_idfs.items(), key=lambda kv: kv[1], reverse=True)[:top_n])
        similarities = []  # targets_doc_tokens中每个文本的相似度
        for target_doc in targets_doc_tokens:
            word2tfidf = self.cal_tf_idf(target_doc)
            sum_tf_idf = sum([word2tfidf[word] for word in keywords])
            similarities.append(sum_tf_idf)
        return similarities