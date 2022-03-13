import tensorflow as tf


# 在transformer架构中
# encoder部分使用的是多头自注意力
# decoder部分使用的是掩码多头自注意力
# 掩码的含义:在decoder部分计算pos位置与其他位置的注意力时,
# 应该忽略pos位置之后的信息,方法是将之后位置的权重设置为一个很小的值
# decoder部分使用掩码自注意力的原因:
# 因为在预测pos位置的输出时,只能依赖pos之前的输入
# 同样,在训练的时候decoder预测第pos位置的词时,只依赖pos之前位置词的输入

# 论文中的描述：
# The input consists of queries and keys of dimension dk,
# and values of dimension dv.
# We compute the dot products of the query with all keys,
# divide each by tf.sqrt(dk),and apply a softmax function to obtain
# the weights on the values.
# 上述描述的公式表示：
# weights = tf.matmul(softmax(tf.matmul(Q,tf.transpose(K))/tf.sqrt(tf.shape(K)[-1])),V)

# shape of q:(batch_size,seq_len_q,dk)
# shape of k:(batch_size,seq_len_k,dk)
# shape of v:(batch_size,seq_len_v,dv)
# shape of scaled dot product q * k的转置/dk开方:(batch_size,seq_len_q,seq_len_k)
# seq_len_k需要与seq_len_v相等
# shape of scaled dot product attention:(batch_size,seq_len_q,dv)
def scaled_dot_product_attention(q, k, v, mask):
    """
    Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.
    Args:
      q: query shape == (..., seq_len_q, depth)
      k: key shape == (..., seq_len_k, depth)
      v: value shape == (..., seq_len_v, depth_v)
      mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.
    Returns:
      output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        # shape of scaled_attention_logits:(batch_size,seq_len_q,seq_len_v)
        # shape of mask:(None,seq_len_q,seq_len_v)
        scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights





if __name__ == '__main__':
    '''
    当多维矩阵进行点乘时,其实点乘的是最后两个维度
    除最后两个维度在维度上满足点乘的条件外,其他维度需要一致
    '''
    # 两个三维矩阵的点乘
    tensor1 = tf.random.normal(shape=(2, 3, 4))
    tensor2 = tf.random.normal(shape=(2, 4, 3))
    matmul = tf.matmul(tensor1, tensor2)
    print(matmul.shape)
    # 一个三维与一个二维矩阵的点乘
    tensor1 = tf.random.normal(shape=(2, 3, 4))
    tensor2 = tf.random.normal(shape=(4, 3))
    matmul = tf.matmul(tensor1, tensor2)
    print(matmul.shape)
