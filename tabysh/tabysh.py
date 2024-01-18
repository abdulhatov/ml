import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from sklearn.preprocessing import LabelEncoder
from project.tabysh.source import (
    texts,
    labels,
    label_mapping
)
label_mapping_inv = {v: k for k, v in label_mapping.items()}
tokenizer = Tokenizer()


def get_tabysh():
    # Преобразование строковых меток в числовые
    label_encoder = LabelEncoder()

    tokenizer.fit_on_texts(texts)

    sequences = tokenizer.texts_to_sequences(texts)

    padded_sequences = pad_sequences(sequences)

    numeric_labels = np.array([label_mapping[label] for label in labels], dtype=np.int32)

    model = Sequential()
    model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=16, input_length=padded_sequences.shape[1]))
    model.add(LSTM(32))
    model.add(Dense(len(label_mapping), activation='softmax'))  # 6 - жондомолор

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # үйрөтүү
    epochs = 20
    history = model.fit(padded_sequences, numeric_labels, epochs=epochs, validation_split=0.30)

    # тактыктын жана катанын маанисин алуу
    train_accuracy = history.history['accuracy']
    test_accuracy = history.history['val_accuracy']
    train_loss = history.history['loss']
    test_loss = history.history['val_loss']

    epochs = range(1, len(train_accuracy) + 1)

    # тактыктын графиги
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_accuracy, 'bo-', label='Training accuracy')
    plt.plot(epochs, test_accuracy, 'ro-', label='Testing accuracy')
    plt.title('Training and Testing Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # катанын графиги
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_loss, 'bo-', label='Training loss')
    plt.plot(epochs, test_loss, 'ro-', label='Testing loss')
    plt.title('Training and Testing Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    #plt.show()

    return model, padded_sequences


def get_tabysh_predict(input_text, padded_sequences, model):
    new_sequences = tokenizer.texts_to_sequences(input_text)
    new_padded_sequences = pad_sequences(new_sequences, maxlen=padded_sequences.shape[1])
    predictions = model.predict(new_padded_sequences)
    predicted_labels = [label_mapping_inv[tf.argmax(prediction).numpy()] for prediction in predictions]
    print(f'Табыш : {input_text[0]+predicted_labels[0]}')



