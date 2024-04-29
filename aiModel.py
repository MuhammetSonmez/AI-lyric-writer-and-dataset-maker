import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import re

SEQ_LENGTH = 128# 100
BATCH_SIZE = 32 # 64
BUFFER_SIZE = 10000 # 10000
VOCAB_SIZE = 0 # 0
EMBEDDING_DIM = 512 # 256 
RNN_UNITS = 1024 # 1024
EPOCHS = 40 # 40
CHECKPOINT_DIR = './training_checkpoints'

def load_data():
    path = r"datasets\rock-test-dataset.txt"
    text = open(path, 'rb').read().decode(encoding='utf-8')
    return text

def preprocess(text):
    text = re.sub(r'[^\x00-\x7F]', '', text)
    vocab = sorted(set(text))
    global VOCAB_SIZE
    VOCAB_SIZE = len(vocab)
    print(f'{len(vocab)} unique characters')

    char2idx = {u: i for i, u in enumerate(vocab)}
    idx2char = np.array(vocab)
    text_as_int = np.array([char2idx[c] for c in text])

    return idx2char, char2idx, text_as_int

def create_dataset(text_as_int):
    char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
    sequences = char_dataset.batch(SEQ_LENGTH + 1, drop_remainder=True)
    dataset = sequences.map(split_input_target)
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
    return dataset

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
        tf.keras.layers.GRU(rnn_units, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model

def train_model(dataset):
    model = build_model(VOCAB_SIZE, EMBEDDING_DIM, RNN_UNITS, BATCH_SIZE)
    model.compile(optimizer='adam', loss=lambda labels, logits: tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True))

    checkpoint_prefix = os.path.join(CHECKPOINT_DIR, 'chkpt_{epoch}')
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True)
    
    """
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix, 
    save_weights_only=True,
    monitor='val_loss',
    save_best_only=True
    )"""

    history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])
    return model, history

def plot_training_loss(history):
    plt.plot(range(1, len(history.history['loss']) + 1), history.history['loss'])
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()

def generate_text(model, start_string, idx2char, char2idx):
    num_generate = 1000
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    text_generated = []
    temperature = 1.0

    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(idx2char[predicted_id])

    return start_string + ''.join(text_generated)

def main(start_string:str, draw_loss:bool = False):
    text = load_data()
    idx2char, char2idx, text_as_int = preprocess(text)
    dataset = create_dataset(text_as_int)
    model, history = train_model(dataset)
    if draw_loss:
        plot_training_loss(history)

    model = build_model(VOCAB_SIZE, EMBEDDING_DIM, RNN_UNITS, batch_size=1)
    model.load_weights(tf.train.latest_checkpoint(CHECKPOINT_DIR))
    model.build(tf.TensorShape([1, None]))

    lyric = generate_text(model, start_string=start_string, idx2char=idx2char, char2idx=char2idx)
    print(lyric)
    return lyric

if __name__ == '__main__':
    main("Every ", True)
