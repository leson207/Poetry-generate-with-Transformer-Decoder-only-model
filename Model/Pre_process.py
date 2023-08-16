import numpy as np
import tensorflow as tf
from keras_preprocessing.sequence import pad_sequences


def process_data(dataset, tokenizer, seq_len):
    input_sequences = []
    for line in dataset['train']['Content']:
        token_list = tokenizer.texts_to_sequences([line])[0]
        length = len(token_list)
        num_seq = int(np.ceil(length / seq_len))
        if num_seq != 1:
            for i in range(num_seq - 1):
                n_gram_sequence = token_list[seq_len * i:seq_len * (i + 1)]
                input_sequences.append(n_gram_sequence)
        if num_seq * seq_len < length:
            n_gram_sequence = token_list[seq_len * (num_seq - 1):]
            input_sequences.append(n_gram_sequence)

    # pad sequences
    xs = np.array(pad_sequences(input_sequences, maxlen=seq_len, padding='pre'))

    split_point = int(len(xs) * 0.7)

    # Split the array into two portions
    train_data = xs[:split_point]
    val_data = xs[split_point:]
    return train_data, val_data


def batched_data(train_data, val_data, BUFFER_SIZE, BATCH_SIZE):
    def train_data_generator():
        for i in train_data:
            yield tf.convert_to_tensor(i[:-1]), tf.convert_to_tensor(i[1:])

    def val_data_generator():
        for i in val_data:
            yield tf.convert_to_tensor(i[:-1]), tf.convert_to_tensor(i[1:])

    def make_batches(ds):
        return (
            ds
            .shuffle(BUFFER_SIZE)
            .batch(BATCH_SIZE)
            .prefetch(buffer_size=tf.data.AUTOTUNE)
        )

    train_data1 = tf.data.Dataset.from_generator(
        train_data_generator,
        output_signature=(
            tf.TensorSpec(shape=(None,), dtype=tf.int64),
            tf.TensorSpec(shape=(None,), dtype=tf.int64)
        )
    )
    val_data1 = tf.data.Dataset.from_generator(
        val_data_generator,
        output_signature=(
            tf.TensorSpec(shape=(None,), dtype=tf.int64),
            tf.TensorSpec(shape=(None,), dtype=tf.int64)
        )
    )
    # Create training and validation set batches.
    train_batches = make_batches(train_data1)
    val_batches = make_batches(val_data1)
    return train_batches, val_batches
