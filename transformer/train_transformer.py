import tensorflow as tf
import tensorflow.python.keras.layers as keras
from get_data import get_train_pairs
from positional_encoding import positional_encoding

# 葡萄牙语转为英语

# 1.读取数据
#   en_tensor shape=(batch_size,en_maxlen)
#   pt_tensor shape=(batch_size,pt_maxlen)
en_tensor, pt_tensor = get_train_pairs()
# 2.语句的最大长度
pt_maxlen = tf.shape(pt_tensor)[1]
en_maxlen = tf.shape(en_tensor)[1]
print('tf.shape(pt_tensor):', tf.shape(pt_tensor))  # [51785   191]
print('tf.shape(en_tensor):', tf.shape(en_tensor))  # [51785   206]
