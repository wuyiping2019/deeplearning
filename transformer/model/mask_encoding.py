import tensorflow as tf


# 该函数用于标识序列中使用PAD填充的位置 用于忽略PAD
def create_padding_mask(seq):
    """
    :param seq: 输入的tensor
    :return: 将输入的tensor中表示PAD填充的位置标识出来 使用1标识 非PAD填充使用0标识
             输出维度[batch,1,1,seq_len]
    """
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    # add extra dimensions to add the padding to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]


# 该函数用于标识序列中应该忽略未来的词权重
# 在scaled dot product attention的计算中scaled_attention_logits += (mask * -1e9)
# 其中scaled_attention_logits的维度是(batch_size,seq_len_q,seq_len_v)
# mask的掩码是统一计算的 所有的样本都一样  一个五维的掩码矩阵如下
# [[0. 1. 1. 1. 1.]
#  [0. 0. 1. 1. 1.]
#  [0. 0. 0. 1. 1.]
#  [0. 0. 0. 0. 1.]
#  [0. 0. 0. 0. 0.]]
# 第一行[0. 1. 1. 1. 1.]表示后面四个词的权重应该忽略 不参与attention的计算
# mask * -1e9之后后面四个权重值很小,进行softmax之后趋近于0,
# 在与V矩阵进行点乘时,对attention的结果不产生影响
def create_look_ahead_mask(size):
    '''
    :param size:
    :return: 输出一个正方形矩阵 矩阵形状[size,size]
             其中对角线以上的位置都是1 其他位置都是0
    '''
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask


if __name__ == '__main__':
    x = tf.constant([[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]])
    mask_encoding = create_padding_mask(x)
    print('seq:', x)
    print('seq padding mask:', mask_encoding)
    temp = create_look_ahead_mask(x.shape[1])
    print('look_ahead_mask:', temp)
