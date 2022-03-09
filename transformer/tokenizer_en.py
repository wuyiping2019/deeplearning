# 自然语言处理中常需要构建读取语料创建词列表 将语句转为对应词在列表中的index
# tensorflow_text中包含多种tokenizers的实现
"""
将sentence转为token-IDs
1.text.BertTokenizer:适用于拉丁语(英语、葡萄牙语)
2.text.WordpieceTokenizer:使用之前需要进行分词,适用于中文、韩语
3.text.SentencepieceTokenizer:to do
"""
# 介绍text.BertTokenizer的基本使用
from init import *

tf.get_logger().setLevel('ERROR')
pwd = pathlib.Path.cwd()
# 1.创建语料
sample = tf.constant(
    [
        ["when you improve searchability ."],
        ["but what if it were active ?"],
        ["but they did n't test for curiosity ."]
    ]
)
# 4.生成tokenizer模型
bert_tokenizer_params = dict(lower_case=True)
reserved_tokens = ["[PAD]", "[UNK]", "[START]", "[END]"]

bert_vocab_args = dict(
    # The target vocabulary size
    vocab_size=8000,
    # Reserved tokens that must be included in the vocabulary
    reserved_tokens=reserved_tokens,
    # Arguments for `text.BertTokenizer`
    bert_tokenizer_params=bert_tokenizer_params,
    # Arguments for `wordpiece_vocab.wordpiece_tokenizer_learner_lib.learn`
    learn_params={},
)
from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab
en_vocab = bert_vocab.bert_vocab_from_dataset(
    sample,
    **bert_vocab_args
)
print(en_vocab)


# 5.将生成token写入文件方便加载回来
def write_vocab_file(filepath, vocab):
    with open(filepath, 'w') as f:
        for token in vocab:
            print(token, file=f)


write_vocab_file('en_vocab.txt', en_vocab)
en_tokenizer = text.BertTokenizer('en_vocab.txt', **bert_tokenizer_params)

