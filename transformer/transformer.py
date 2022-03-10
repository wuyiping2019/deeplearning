from get_data import pt_words_ids, en_words_ids
from positional_encoding import positional_encoding
import tensorflow as tf
import tensorflow.python.keras.layers as keras

# 葡萄牙语转为英语
maxlen = max([len(line) for line in pt_words_ids]) # 191
embedding_size = 300

# 数据pad
import tf.python.keras.preprocessing.sequence.pad_sequences







