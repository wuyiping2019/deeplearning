from functools import reduce


class SpaceTokenizer:
    def __init__(self, vocab_path, separator=" ", prefix_token="[START]", suffix_token="[END]"):
        self.vocab_path = vocab_path
        self.word2id = None
        self.id2word = None
        self.flag = False
        self.separator = separator
        self.prefix_token = prefix_token
        self.suffix_token = suffix_token

    def init(self):
        """
        初始化word2id和id2word
        :return:
        """
        if not (self.word2id and self.id2word):
            with open(self.vocab_path, 'r', encoding='utf-8') as f:
                words = f.read().split('\n')
            self.word2id = {word: index for index, word in enumerate(words)}
            self.id2word = {index: word for index, word in enumerate(words)}

    def single_tokenize(self, sentence: str):
        """
        对输入的字符串进行分词
        :param sentence: 输入的字符串
        :param separator: 分隔符
        :param prefix_token: 分词列表前面填充元素
        :param suffix_token: 分词列表后面填充元素
        :return:
        """
        # 如果没有初始化word2id和id2word,则进行初始化
        if not self.flag:
            self.init()
            self.flag = True
        words = sentence.split(self.separator)
        # 避免解析的句子中存在不在单词文件中的单词
        for index, word in enumerate(words):
            if word not in self.word2id.keys():
                words[index] = '[UNK]'
        if self.prefix_token != '' and self.prefix_token is not None:
            words = [self.prefix_token] + words
        if self.suffix_token != '' and self.suffix_token is not None:
            words = words + [self.suffix_token]
        return words

    def batch_tokenize(self, sentences):
        return [self.single_tokenize(sentence) for sentence in sentences]

    def sentence2ids(self, sentence):
        return [self.word2id[word] for word in self.single_tokenize(sentence)]

    def sentences2ids(self, sentences):
        return [self.sentence2ids(sentence) for sentence in sentences]

    def ids2sentence(self, sentence_ids):
        return reduce(lambda x, y: x + self.separator + y, [self.id2word[id] for id in sentence_ids], '')

    def ids2sentences(self, sentences_ids):
        return [self.ids2sentence(sentence_ids) for sentence_ids in sentences_ids]
