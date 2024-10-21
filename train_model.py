import pandas as pd
import numpy as np
import tensorflow as tf
import re
from keras.callbacks import EarlyStopping
from gensim.models import KeyedVectors

import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Embedding, GRU, Dense, Dropout, Bidirectional

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

# Load pre-trained Word2Vec embeddings
word2vec_model = KeyedVectors.load_word2vec_format('models/GoogleNews-vectors-negative300.bin', binary=True)

# Load and preprocess the data
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path, sep=',')
    df['label'] = df['label'].map({'hate': 1, 'noHate': 0})
    df['text'] = df['content'].apply(preprocess_text)
    return df['text'].values, df['label'].values

# Text preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])
    return text

# Tokenize and pad sequences
def tokenize_and_pad(texts, max_words=10000, max_len=100, tokenizer=None):
    """
    Tokenize and pad the input texts.

    Args:
        texts (list): List of input texts to be tokenized and padded.
        max_words (int, optional): Maximum number of words to keep in the vocabulary. 
                                   Words beyond this limit will be treated as out-of-vocabulary (OOV).
                                   Defaults to 10000.
        max_len (int, optional): Maximum length of each sequence. Sequences longer than this will be truncated,
                                 and shorter sequences will be padded. Defaults to 100.

    Returns:
        tuple: A tuple containing:
            - padded_sequences (numpy.ndarray): The tokenized and padded sequences.
            - tokenizer (Tokenizer): The fitted Tokenizer object.
    """
    if not isinstance(tokenizer, Tokenizer):
        raise TypeError("tokenizer must be an instance of keras.preprocessing.text.Tokenizer")
    
    if tokenizer is None:
        tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
        tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    return padded_sequences, tokenizer

# Create embedding matrix
def create_embedding_matrix(word_index, embedding_dim=300):
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        if word in word2vec_model.key_to_index:
            embedding_matrix[i] = word2vec_model[word]
    return embedding_matrix


def build_model(vocab_size, embedding_matrix, max_len=100):
    """
    Build and compile the hate speech classification model.

    This function creates a Sequential model with an Embedding layer initialized
    with pre-trained word embeddings, followed by Bidirectional GRU layers,
    Dense layers, and Dropout for regularization. The model is designed for
    binary classification of hate speech.

    Parameters:
    vocab_size (int): Size of the vocabulary (number of unique words).
    embedding_matrix (numpy.ndarray): Pre-trained word embeddings matrix.
    max_len (int, optional): Maximum length of input sequences. Defaults to 100.

    Returns:
    keras.models.Sequential: Compiled Keras model ready for training.

    The model architecture:
    1. Embedding layer (non-trainable, initialized with pre-trained embeddings)
    2. Bidirectional GRU layer (64 units, returning sequences)
    3. Bidirectional GRU layer (32 units)
    4. Dense layer (64 units, ReLU activation)
    5. Dropout layer (50% dropout rate)
    6. Output Dense layer (1 unit, sigmoid activation)

    The model is compiled with Adam optimizer and binary crossentropy loss.
    """
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_matrix.shape[1], 
                        weights=[embedding_matrix], 
                        input_length=max_len, 
                        trainable=False))
    model.add(Bidirectional(GRU(64, return_sequences=True)))
    model.add(Bidirectional(GRU(32)))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Main function
def main():
    # Load and preprocess data
    X, y = load_and_preprocess_data('data/modified_dataset/hate_speech_train_dataset.csv')
    
    # Tokenize and pad sequences
    X_padded, tokenizer = tokenize_and_pad(X)
    
    # Split the data
    X_train_padded, X_val_padded, y_train, y_val = train_test_split(
        X_padded, y, test_size=0.2, random_state=42
    )
    
    # Create embedding matrix
    embedding_matrix = create_embedding_matrix(tokenizer.word_index)
    
    # Build the model
    vocab_size = len(tokenizer.word_index) + 1
    model = build_model(vocab_size, embedding_matrix, max_len=X_train_padded.shape[1])
    
    # Early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    # Train the model
    history = model.fit(
        X_train_padded, y_train,
        epochs=20,
        validation_data=(X_val_padded, y_val),
        batch_size=32,
        callbacks=[early_stopping]
    )
    
    # Save the model
    model.save('models/hate_speech_model.h5')
    
    # Save the tokenizer
    import pickle
    with open('models/tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()
