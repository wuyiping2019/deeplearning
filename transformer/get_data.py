import tensorflow_datasets as tf_ds
from tokenizer.SpaceTokenizer import SpaceTokenizer
import tensorflow as tf

def download_pt2en_corpus():
    examples, metadata = tf_ds.load('ted_hrlr_translate/pt_to_en', with_info=True, as_supervised=True)
    train_examples, val_examples, test_examples = examples['train'], examples['validation'], examples['test']
    f_en = open('data/en_corpus.txt', 'w', encoding='utf-8')
    f_pt = open('data/pt_corpus.txt', 'w', encoding='utf-8')
    for pt_examples, en_examples in train_examples.batch(1).take(100000):
        for index, pt in enumerate(pt_examples.numpy()):
            if index == len(pt_examples.numpy()):
                f_pt.write(pt.decode('utf-8'))
            else:
                f_pt.write(pt.decode('utf-8') + '\n')
        for index, en in enumerate(en_examples.numpy()):
            if index == len(en_examples.numpy()):
                f_en.write(en.decode('utf-8'))
            else:
                f_en.write(en.decode('utf-8') + '\n')
    f_pt.close()
    f_en.close()


def write_pt2en_vocabs():
    f_en = open('data/en_corpus.txt', 'r', encoding='utf-8')
    f_pt = open('data/pt_corpus.txt', 'r', encoding='utf-8')
    en_vocab_set = set()
    pt_vocab_set = set()
    en_sample_list = f_en.read().strip().split('\n')
    pt_sample_list = f_pt.read().strip().split('\n')
    for line in en_sample_list:
        for word in line.split(' '):
            en_vocab_set.add(word)
    for line in pt_sample_list:
        for word in line.split(' '):
            pt_vocab_set.add(word)
    f_en.close()
    f_pt.close()
    reserved_tokens = ["[PAD]", "[UNK]", "[START]", "[END]"]
    pt_vocabs = reserved_tokens + sorted(list(pt_vocab_set))
    en_vocabs = reserved_tokens + sorted(list(en_vocab_set))
    f_en = open('data/en_vocabs.txt', 'w', encoding='utf-8')
    f_pt = open('data/pt_vocabs.txt', 'w', encoding='utf-8')
    for index, word in enumerate(pt_vocabs):
        if index == len(pt_vocabs):
            f_pt.write(word)
        else:
            f_pt.write(word + '\n')
    for index, word in enumerate(en_vocabs):
        if index == len(en_vocabs):
            f_en.write(word)
        else:
            f_en.write(word + '\n')
    f_en.close()
    f_pt.close()
    return pt_sample_list, en_sample_list


def get_train_pairs():
    pt_sample_list, en_sample_list = write_pt2en_vocabs()
    en_tokenizer = SpaceTokenizer('data/en_vocabs.txt', separator=' ', prefix_token="[START]", suffix_token="[END]")
    pt_tokenizer = SpaceTokenizer('data/pt_vocabs.txt', separator=' ', prefix_token=None, suffix_token=None)
    en_words_ids = en_tokenizer.sentences2ids(en_sample_list)
    pt_words_ids = pt_tokenizer.sentences2ids(pt_sample_list)
    en_tensor = tf.ragged.constant(en_words_ids).to_tensor()
    pt_tensor = tf.ragged.constant(pt_words_ids).to_tensor()
    return en_tensor, pt_tensor


if __name__ == '__main__':
    en_words_ids, pt_words_ids = get_tokenized_ids()
    print(en_words_ids[-1])
    print(pt_words_ids[-1])
