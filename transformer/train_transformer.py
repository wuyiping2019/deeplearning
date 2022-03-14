from transformer.model.transformer import Transformer
import tensorflow_datasets as tf_ds
import time
# 需要导入 否则会报KeyError: 'CaseFoldUTF8'
import tensorflow_text
from transformer.model.config import *

# #########################样本数据处理##########################

BUFFER_SIZE = 20000
BATCH_SIZE = 64


def tokenize_pairs(pt, en):
    pt = tokenizers.pt.tokenize(pt).to_tensor()
    en = tokenizers.en.tokenize(en).to_tensor()
    return pt, en


def make_batches(ds):
    return (
        ds.cache()
            .shuffle(BUFFER_SIZE)
            .batch(BATCH_SIZE)
            .map(tokenize_pairs, num_parallel_calls=tf.data.AUTOTUNE)
            .prefetch(tf.data.AUTOTUNE)
    )


examples, metadata = tf_ds.load('ted_hrlr_translate/pt_to_en',
                                with_info=True,
                                as_supervised=True)
train_examples, val_examples = examples['train'], examples['validation']
for pt_examples, en_examples in train_examples.batch(3).take(1):
    for pt in pt_examples.numpy():
        print(pt.decode('utf-8'))
    for en in en_examples.numpy():
        print(en.decode('utf-8'))
model_name = "ted_hrlr_translate_pt_en_converter"
tf.keras.utils.get_file(
    f"{model_name}.zip",
    f"https://storage.googleapis.com/download.tensorflow.org/models/{model_name}.zip",
    cache_dir='.', cache_subdir='', extract=True
)
tokenizers = tf.saved_model.load('ted_hrlr_translate_pt_en_converter')
train_batches = make_batches(train_examples)
val_batches = make_batches(val_examples)
# #########################创建并训练模型##########################
transformer = Transformer(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    dff=dff,
    input_vocab_size=tokenizers.pt.get_vocab_size().numpy(),
    target_vocab_size=tokenizers.en.get_vocab_size().numpy(),
    pe_input=1000,
    pe_target=1000,
    rate=dropout_rate)
checkpoint_path = "./checkpoints/train"
learning_rate = CustomSchedule(d_model)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,epsilon=1e-9)
temp_learning_rate_schedule = CustomSchedule(d_model)
ckpt = tf.train.Checkpoint(transformer=transformer,optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!!')
EPOCHS = 20
# The @tf.function trace-compiles train_step into a TF graph for faster
# execution. The function specializes to the precise shape of the argument
# tensors. To avoid re-tracing due to the variable sequence lengths or variable
# batch sizes (the last batch is smaller), use input_signature to specify
# more generic shapes.

# 记录train_loss
train_loss = tf.keras.metrics.Mean(name='train_loss')
# 记录train_accuracy
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')
train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]
@tf.function(input_signature=train_step_signature)
def train_step(inp, tar):
    # 错位处理 其实该处理过程可以放在数据处理过程中,然后在这个函数放三个形参 inp\tar_inp\tar_real
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]
    with tf.GradientTape() as tape:
        predictions, _ = transformer([inp, tar_inp],training=True)
        loss = loss_function(tar_real, predictions)
    # 计算梯度
    gradients = tape.gradient(loss, transformer.trainable_variables)
    # 更新训练参数
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
    # 记录损失和准确率
    train_loss(loss)
    train_accuracy(accuracy_function(tar_real, predictions))


for epoch in range(EPOCHS):
    # 记录批次开始时间 重置记录的损失和准确率
    start = time.time()
    train_loss.reset_states()
    train_accuracy.reset_states()
    # inp -> portuguese, tar -> english
    # 遍历数据集中的batch进行训练
    for (batch, (inp, tar)) in enumerate(train_batches):
        train_step(inp, tar)
        if batch % 50 == 0:
            print(f'Epoch {epoch + 1} Batch {batch} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')
    if (epoch + 1) % 5 == 0:
        ckpt_save_path = ckpt_manager.save()
        print(f'Saving checkpoint for epoch {epoch + 1} at {ckpt_save_path}')
    print(f'Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')
    print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')