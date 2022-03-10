from init import *
import tensorflow as tf
sample = \
    [
        "when you improve searchability .",
        "but what if it were active ?",
        "but they did n't test for curiosity ."
    ]
"""
tf.data.TextLineDataset(
    filenames, 
    compression_type=None, 
    buffer_size=None, 
    num_parallel_reads=None,
    name=None
)
"""
with open('TextLineDataset.txt', 'w', encoding='utf-8') as f:
    for line in sample:
        f.write(line + '\n')
dataset = tf.data.TextLineDataset(filenames='TextLineDataset.txt')
for line in dataset:
    print(line)
