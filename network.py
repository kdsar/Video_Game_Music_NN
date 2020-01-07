from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
# from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Embedding


from network_utils import *
from parser import *

import os

num_layers = 6
d_model = 512
dff = 2048
num_heads = 8
dropout_rate = 0.1

EPOCHS = 100
context = 50

CHECKPOINT_DIR = './training_checkpoints'
CHECKPOINT_PREFIX = os.path.join(CHECKPOINT_DIR, "ckpt")


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding,
                                                self.d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]

        self.dropout = Dropout(rate)

    def call(self, x, training, mask):

        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
                 maximum_position_encoding, rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(
            maximum_position_encoding, d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]
        self.dropout = Dropout(rate)

    def call(self, x, enc_output, training,
             look_ahead_mask, padding_mask):

        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                                   look_ahead_mask, padding_mask)

            attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights


class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 target_vocab_size, pe_input, pe_target, rate=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, dff,
                               input_vocab_size, pe_input, rate)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff,
                               target_vocab_size, pe_target, rate)

        self.final_layer = Dense(target_vocab_size)

    def call(self, inp, tar, training, enc_padding_mask,
             look_ahead_mask, dec_padding_mask):

        # (batch_size, inp_seq_len, d_model)
        enc_output = self.encoder(inp, training, enc_padding_mask)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            tar, enc_output, training, look_ahead_mask, dec_padding_mask)

        # (batch_size, tar_seq_len, target_vocab_size)
        final_output = self.final_layer(dec_output)

        return final_output, attention_weights


train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]
train_loss = get_train_loss()
train_accuracy = get_train_accuracy()
learning_rate = CustomSchedule(d_model)
tokenizer = get_tokenizer()
vocab_size = len(tokenizer.index_word) + 1
optimizer = Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                 epsilon=1e-9)
transformer = Transformer(num_layers, d_model, num_heads, dff,
                          vocab_size, vocab_size,
                          pe_input=context,
                          pe_target=context,
                          rate=dropout_rate)

checkpoint = tf.train.Checkpoint(transformer=transformer,
                                 optimizer=optimizer)


def train_step(inp):
    inp = inp['encoded_note_string']
    tar_inp = inp[:, :-1]
    tar_real = inp[:, 1:]

    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
        inp, tar_inp)

    with tf.GradientTape() as tape:
        predictions, _ = transformer(inp, tar_inp,
                                     True,
                                     enc_padding_mask,
                                     combined_mask,
                                     dec_padding_mask)

        loss = loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    train_loss(loss)
    train_accuracy(tar_real, predictions)


def train(dataset):
    for epoch in range(EPOCHS):

        train_loss.reset_states()
        train_accuracy.reset_states()

        for (batch, inp) in enumerate(dataset):
            train_step(inp)

            if batch % 50 == 0:
                print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                    epoch + 1, batch, train_loss.result(), train_accuracy.result()))

        if (epoch + 1) % 20 == 0:
            checkpoint.save(file_prefix=CHECKPOINT_PREFIX)
            print('Saving checkpoint for epoch {}'.format(epoch + 1))

        print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1,
                                                            train_loss.result(),
                                                            train_accuracy.result()))


def evaluate(inp_sentence):
    '''
    Uses the current tranformer to predict the next steps of the given input sequence
    '''
    tokenized_input = get_tensor_from_tokenizer_and_corpus(
        tokenizer, [inp_sentence])
    start_token, end_token = get_tensor_from_tokenizer_and_corpus(
        tokenizer, ['start', 'finish'])
    for i in range(30):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
            tokenized_input, tokenized_input)

        predictions, attention_weights = transformer(tokenized_input,
                                                     tokenized_input,
                                                     False,
                                                     enc_padding_mask,
                                                     combined_mask,
                                                     dec_padding_mask)

        predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        # return the result if the predicted_id is equal to the end token
        if predicted_id == end_token:
            return tf.squeeze(tokenized_input, axis=0), attention_weights

        # concatentate the predicted_id to the output which is given to the decoder
        # as its input.
        tokenized_input = tf.concat(
            [tokenized_input, predicted_id], axis=-1)

    return tf.squeeze(tokenized_input, axis=0), attention_weights


def run():
    prepare_dataset()
    train(open_dataset(64))
    sentence, weights = evaluate('start tempo:150')
    print(sentence)
