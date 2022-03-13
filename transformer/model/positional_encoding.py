import numpy as np
import tensorflow as tf


# 代码实现来自tensorflow官网
def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    """

    :param position: 表示位置的长度 = maxlen
    :param d_model: 表示输出的位置编码向量长度
                    一般使用词向量与该词的位置向量进行点加来表示在词向量中添加了位置信息
                    因此d_model = embedding_size
    :return:
            输出的是位置矩阵 shape [position, d_model] = [maxlen, embedding_size]
            其中每一行表示对应行索引位置词的位置信息
    :use:
         当获取位置编码矩阵之后,需要与样本数据进行相加操作
         位置矩阵shape [1, maxlen, embedding_size]
         样本数据矩阵shape [batch_size,maxlen,embedding_size]
         [maxlen,embedding_size] + [batch_size, maxlen, embedding_size] 会进行广播
    """
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)


if __name__ == '__main__':
    # 查看位置编码输出tensorflow的形状
    # （1,maxlen,d_model）
    maxlen, d_model = 2048, 512
    pos_encoding = positional_encoding(maxlen, d_model)
    print(pos_encoding.shape)  # (1, 2048, 512)
