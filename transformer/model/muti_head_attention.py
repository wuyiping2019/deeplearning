import tensorflow as tf
from transformer.model.scaled_dot_product_attention import scaled_dot_product_attention


# MutiHeadAtttinon就是多个Attention的输出的拼接
# Transformer结构中encoder和decoder都是堆叠的层,一个层的输出是另一个层的输入
# num_head是指单层encoder或decoder中Attention的个数
# 论文中设置dk=dv
class MultiHeadAttention(tf.keras.layers.Layer):

    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        # num_heads = 8 头的个数
        self.num_heads = num_heads
        # d_model = 512 与num_heads * dv相等
        self.d_model = d_model
        assert d_model % self.num_heads == 0
        # dk=dv=512/8=64
        self.depth = d_model // self.num_heads
        # dq=dk=dv
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """
        拆分每个scaled dot product对应的Q\k\V
        Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        # shape (batch_size,seq_len,num_heads,depth)
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        # 转置 shape (batch_size,num_heads,seq_len,depth)
        # 转置的原因：Q\K\V的计算过程中使用矩阵点乘
        #           应该每个scaled dot product attention自己的Q\K\V进行点乘操作
        #           四维矩阵点乘的是最后两个维度 前面的两个维度需要相等
        #           前面两个维度是(batch_size,num_head)
        #           后面两个维度是(seq_len,depth)
        # 其实就是方便使用四维矩阵进行多头自注意力的计算
        return tf.transpose(x, perm=[0, 2, 1, 3])

    # 注意：
    # 在transformer架构中,encoder和decoder堆叠块的输入是一个矩阵
    # 该矩阵分别使用三个Dense层输出Q\K\V 然后计算Attention
    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)
        # 需要进行多头自注意力的多个头的attention进行拼接操作
        # 先转置回去为(batch_size, seq_len_q, num_heads, depth)
        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])

        # 然后进行reshape为(batch_size, seq_len_q, d_model)
        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights
