# 内容来自tensorflow官网
# Transformer model for language understanding
# pip install tensorflow_datasets
# pip install -U "tensorflow-text==2.8.*"

import collections
import logging
import os
import pathlib
import re
import string
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

import tensorflow_datasets as tfds
import tensorflow_text as text
import tensorflow as tf

logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings

examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en',
                               with_info=True,
                               as_supervised=True)
# type(examples): <class 'dict'>
print('type(examples):', type(examples))
# type(metadata): <class 'tensorflow_datasets.core.dataset_info.DatasetInfo'>
print('type(metadata):', type(metadata))
# examples: dict_keys(['train', 'validation', 'test'])
print('examples.keys():', examples.keys())
# print('metadata:', metadata)

# 查看数据
train_examples, val_examples, test_examples = examples['train'], examples['validation'], examples['test']
for pt_examples, en_examples in train_examples.batch(3).take(1):
    '''
    e quando melhoramos a procura , tiramos a única vantagem da impressão , que é a serendipidade .
    mas e se estes fatores fossem ativos ?
    mas eles não tinham a curiosidade de me testar .
    '''
    for pt in pt_examples.numpy():
        print(pt.decode('utf-8'))
    print()
    '''
    and when you improve searchability , you actually take away the one advantage of print , which is serendipity .
    but what if it were active ?
    but they did n't test for curiosity .
    '''
    for en in en_examples.numpy():
        print(en.decode('utf-8'))

model_name = "ted_hrlr_translate_pt_en_converter"
tf.keras.utils.get_file(
    f"{model_name}.zip",
    f"https://storage.googleapis.com/download.tensorflow.org/models/{model_name}.zip",
    cache_dir='.', cache_subdir='', extract=True
)

tokenizers = tf.saved_model.load(model_name)
print([item for item in dir(tokenizers.en) if not item.startswith('_')])
for en in en_examples.numpy():
    print(en.decode('utf-8'))

encoded = tokenizers.en.tokenize(en_examples)

for row in encoded.to_list():
    print(row)

round_trip = tokenizers.en.detokenize(encoded)
for line in round_trip.numpy():
    print(line.decode('utf-8'))

tokens = tokenizers.en.lookup(encoded)
print(tokens)
