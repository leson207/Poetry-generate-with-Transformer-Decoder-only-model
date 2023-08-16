import tensorflow as tf
from datasets import load_dataset
from transformers import AutoTokenizer
from Decoder_with_pretrained_embedding import Transformer
from Post_process_with_gpt2_tokenizer import Beam_search
from Pre_process_with_gpt2_tokenizer import process_data, batched_data

class ExportTranslator(tf.Module):
    def __init__(self, translator):
        super().__init__()
        self.translator = translator

    @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
    def __call__(self, sentence):
        (result) = self.translator(sentence, max_length=seq_len, num_gen=20)

        return result


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.initial_learning_rate = 0.01
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        return {"initial_learning_rate": self.initial_learning_rate}


def masked_loss(label, pred):
    mask = label != 0
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    loss = loss_object(label, pred)

    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask

    loss = tf.reduce_sum(loss) / tf.reduce_sum(mask)
    return loss


def masked_accuracy(label, pred):
    pred = tf.argmax(pred, axis=2)
    label = tf.cast(label, pred.dtype)
    match = label == pred

    mask = label != 0

    match = match & mask

    match = tf.cast(match, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(match) / tf.reduce_sum(mask)


seq_len = 128
BUFFER_SIZE = 20000
BATCH_SIZE = 64
num_layers = 4
d_model = 128
dff = 512
num_heads = 8
dropout_rate = 0.1

dataset = load_dataset("csv", data_files="Pushkin.csv", encoding='latin-1')

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = 0
total_word=len(tokenizer)
seq_len=128
train_data, val_data = process_data(dataset, tokenizer, seq_len)

train_batches, val_batches = batched_data(train_data, val_data, BUFFER_SIZE, BATCH_SIZE)

transformer = Transformer(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    dff=dff,
    vocab_size=total_word,
    dropout_rate=dropout_rate)

learning_rate = CustomSchedule(d_model)

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)

transformer.compile(
    loss=masked_loss,
    optimizer=optimizer,
    metrics=[masked_accuracy])

translator = Beam_search(tokenizer, transformer)
# translator = Translator(tokenizer, transformer)
sentence = ['I love you', 'i like you']

translator(tf.constant(sentence))

# translator = ExportTranslator(translator)
#
# translator('i love you').numpy()
# tf.saved_model.save(translator, export_dir='translator')
# reloaded = tf.saved_model.load('translator')
# reloaded.translator('i love you').numpy()
# transformer.save('/content')
