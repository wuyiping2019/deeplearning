import tensorflow as tf

"""
论文位置
3.3 Position-wise Feed-Forward Networks
In addition to attention sub-layers,
each of the layers in our encoder and decoder contains a fully connected 
feed-forward network,which is applied to each position separately and identically.
This consists of two linear transformations with a ReLU activation in between.

transformer堆叠块的每个子层都有由两个前馈层组成
论文中设置dff=2048
先映射到2048维的向量 再映射回512维的向量 保持进入块的维度与块输出的维度相同
"""


def point_wise_feed_forward_network(d_model, dff):
    '''
    transformer架构中
    :param d_model:
    :param dff:
    :return:
    '''
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])
