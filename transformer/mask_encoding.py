import tensorflow as tf


def create_padding_mask(seq):
    """
    :param seq: 输入的tensor
    :return: 将输入的tensor中表示PAD填充的位置标识出来 使用1标识 非PAD填充使用0标识
             输出维度[batch,1,1,seq_len]
    """
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(size):
    '''
    :param size:
    :return: 输出一个正方形矩阵 矩阵形状[size,size]
             其中对角线以上的位置都是1 其他位置都是0
    '''
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


if __name__ == '__main__':
    x = tf.constant([[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]])
    mask_encoding = create_padding_mask(x)
    print('seq:', x)
    print('seq padding mask:', mask_encoding)
    x = tf.random.uniform((1, 3))
    temp = create_look_ahead_mask(x.shape[1])
    print('look_ahead_mask:', temp)
