import numpy as np
import tensorflow as tf
from keras_preprocessing.sequence import pad_sequences


class Normal_search(tf.Module):
    def __init__(self, tokenizer, transformer):
        super().__init__()
        self.tokenizer = tokenizer
        self.transformer = transformer

    def __call__(self, sentences, max_length=128, num_gen=20):

        sentences = sentences.numpy()
        if type(sentences) == np.ndarray:
            sentences = [s.decode() for s in sentences]
        else:
            sentences = sentences.decode()

        if type(sentences) == str:
            sentences = [sentences]

        tokens = [self.tokenizer.encode(sentence, add_special_tokens=False) for sentence in sentences]

        for i in range(num_gen):
            inputs = pad_sequences(tokens, maxlen=max_length, padding='pre', truncating='pre')

            predictions = self.transformer(tf.convert_to_tensor(inputs), training=False)

            predictions = predictions[:, -1:, :]  # Shape `(batch_size, 1, vocab_size)`.

            predicted_id = tf.argmax(predictions, axis=-1)

            for j in range(len(sentences)):
                tokens[j] = tokens[j] + [predicted_id.numpy()[j][0]]

        return [self.tokenizer.decode(token, add_special_tokens=False) for token in tokens]


class Beam_search(tf.Module):
    def __init__(self, tokenizer, transformer):
        super().__init__()
        self.tokenizer = tokenizer
        self.transformer = transformer

    def __call__(self, sentences, max_length=128, num_gen=20):

        sentences = sentences.numpy()
        if type(sentences) == np.ndarray:
            sentences = [s.decode() for s in sentences]
        else:
            sentences = sentences.decode()

        if type(sentences) == str:
            sentences = [sentences]

        band_width = 3
        tokens = [self.tokenizer.encode(sentence, add_special_tokens=False) for sentence in sentences]

        cur_seq = []
        cur_prob = []
        cur_len = []
        batch_size = len(sentences)
        for j in range(batch_size):
            cur_seq = cur_seq + [tokens[j]] * band_width

            cur_prob = cur_prob + [0.] * band_width

            cur_len = cur_len + [0] * band_width

        for i in range(num_gen):
            inputs = pad_sequences(cur_seq, maxlen=max_length, padding='pre', truncating='pre')

            predictions = self.transformer(tf.convert_to_tensor(inputs), training=False)

            predictions = predictions[:, -1:, :]  # Shape `(batch_size*band_width, 1, vocab_size)`.

            # predicted_id = tf.argmax(predictions, axis=-1)

            _, predicted_id = tf.math.top_k(predictions, k=band_width)

            for j in range(0, batch_size * band_width, band_width):
                candidate_list = []
                for k in range(j, j + band_width):
                    for t in range(band_width):
                        idx = predicted_id.numpy()[k][0][t]
                        next_list = cur_seq[k] + [idx]
                        next_prob = cur_prob[k] + np.log(predictions.numpy()[k][0][idx])
                        next_len = cur_len[k] + 1
                        mean_pob = next_prob / next_len

                        candidate_list = candidate_list + [(next_list, next_prob, next_len, mean_pob)]

                sorted_list = sorted(candidate_list, key=lambda x: x[-1], reverse=True)
                for k in range(j, j + band_width):
                    cur_seq[k] = sorted_list[k - j][0]
                    cur_prob[k] = sorted_list[k - j][1]
                    cur_len[k] = sorted_list[k - j][2]

        gen_text = []

        for i in range(0, batch_size * band_width, band_width):
            tmp = cur_prob[i:i + band_width]
            idx = tmp.index(max(tmp))
            gen_text = gen_text + [cur_seq[i + idx]]
        return [self.tokenizer.decode(gen_text_element) for gen_text_element in gen_text]
