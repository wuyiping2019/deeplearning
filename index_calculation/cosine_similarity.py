import tensorflow as tf


# 计算两个向量矩阵的余弦
# input:tensor1 shape(A,)
def tensor_cosine_similarity(tensor1: tf.Tensor, tensor2: tf.Tensor):
    """
    计算tensor1中每一个样本与tensor2中所有样本的余弦相似度
    tensor1和tensor2的batch_size可以不相同
    tensor1和tensor2的vector_dim必须相同
    Args:
        tensor1: shape (tensor1_batch_size,vector_dim)  每一行代表一个样本
        tensor2: shape (tensor2_batch_size,vector_dim)  每一行代表一个样本
    Returns: shape (tensor1_batch_size,tensor2_batch_size)
             每一行代表tensor1中的一个样本与tensor2中各个样本的余弦距离
    """
    # 计算内积
    matmul = tf.matmul(tensor1, tf.transpose(tensor2))  # (tensor1_batch_size,tensor2_batch_size)
    print(tf.shape(matmul))
    # 计算tensor1的欧式距离
    tensor1_norm = tf.sqrt(tf.reduce_sum(tf.square(tensor1), -1))[:, tf.newaxis]  # (tensor1_batch_size,1)
    print(tf.shape(tensor1_norm))
    # 计算tensor2的欧式距离
    tensor2_norm = tf.sqrt(tf.reduce_sum(tf.square(tensor2), -1))[tf.newaxis, :]  # (1,tensor2_batch_size)
    print(tf.shape(tensor2_norm))
    return tf.divide(matmul, (tf.matmul(tensor1_norm, tensor2_norm)))


if __name__ == '__main__':
    tensor1 = tf.cast(tf.constant([[1, 2, 3], [0, 0, 0]]), tf.float32)
    tensor2 = tf.cast(tf.constant([[2, 3, 3], [0, 2, 4], [0, 0, 0]]), tf.float32)
    print(tensor_cosine_similarity(tensor1, tensor2))
