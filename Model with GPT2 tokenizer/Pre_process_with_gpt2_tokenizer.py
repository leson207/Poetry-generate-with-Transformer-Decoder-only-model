import tensorflow as tf


def process_data(dataset, tokenizer, seq_len):
    xs = []
    for element in dataset['train']['Content']:
        token = tokenizer(
            element,
            truncation=True,
            max_length=seq_len,
            return_overflowing_tokens=True,
            return_length=True)
        for length, input_ids in zip(token['length'], token['input_ids']):
            if length == seq_len:
                xs.append(input_ids)
            else:
                input_ids = [0] * (seq_len - length) + input_ids[:length]  # Add padding at the beginning
                xs.append(input_ids)

    split_point = int(len(xs) * 0.9)

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
