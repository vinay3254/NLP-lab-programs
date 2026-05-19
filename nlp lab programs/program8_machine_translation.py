# ============================================================
# Program 8:
# Implement the machine translation application of NLP where it
# needs to train a machine translation model for a language with
# limited parallel corpora. Investigate and incorporate techniques
# to improve performance in low-resource scenarios.
#
# Approach:
#   - Use a small English-Hindi parallel corpus (low-resource)
#   - Apply data augmentation (sentence reversal) to expand data
#   - Build an Encoder-Decoder LSTM model using TensorFlow/Keras
#   - Train the model and test translation on sample input
# ============================================================

# ============================================
# 1. IMPORT LIBRARIES
# ============================================
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ============================================
# 2. CREATE LOW-RESOURCE PARALLEL DATASET
# ============================================
eng_sentences = [
    "hello",
    "how are you",
    "i am fine",
    "what is your name",
    "good morning"
]

hin_sentences = [
    "namaste",
    "aap kaise hain",
    "main theek hoon",
    "aapka naam kya hai",
    "suprabhat"
]

# ============================================
# 3. DATA AUGMENTATION (LOW-RESOURCE TECHNIQUE)
# ============================================
aug_eng = []
aug_hin = []

for e, h in zip(eng_sentences, hin_sentences):
    aug_eng.append(e[::-1])
    aug_hin.append(h[::-1])

eng_sentences += aug_eng
hin_sentences += aug_hin

# ============================================
# 4. TOKENIZATION
# ============================================
eng_tokenizer = Tokenizer()
eng_tokenizer.fit_on_texts(eng_sentences)

hin_tokenizer = Tokenizer()
hin_tokenizer.fit_on_texts(hin_sentences)

eng_seq = eng_tokenizer.texts_to_sequences(eng_sentences)
hin_seq = hin_tokenizer.texts_to_sequences(hin_sentences)

# ============================================
# 5. PADDING SEQUENCES
# ============================================
max_len_eng = max(len(seq) for seq in eng_seq)
max_len_hin = max(len(seq) for seq in hin_seq)

eng_seq = pad_sequences(eng_seq, maxlen=max_len_eng, padding='post')
hin_seq = pad_sequences(hin_seq, maxlen=max_len_hin, padding='post')

# ============================================
# 6. BUILD ENCODER-DECODER MODEL
# ============================================
latent_dim = 64

# Encoder
encoder_inputs = Input(shape=(None,))
enc_emb = tf.keras.layers.Embedding(
    len(eng_tokenizer.word_index) + 1, latent_dim
)(encoder_inputs)
encoder_lstm = LSTM(latent_dim, return_state=True)
_, state_h, state_c = encoder_lstm(enc_emb)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = Input(shape=(None,))
dec_emb = tf.keras.layers.Embedding(
    len(hin_tokenizer.word_index) + 1, latent_dim
)(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)
decoder_dense = Dense(len(hin_tokenizer.word_index) + 1, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# ============================================
# 7. PREPARE TARGET DATA
# ============================================
hin_seq_expanded = np.expand_dims(hin_seq, -1)

# ============================================
# 8. TRAIN MODEL
# ============================================
model.fit(
    [eng_seq, hin_seq],
    hin_seq_expanded,
    epochs=100,
    verbose=0
)
print("Model training complete.")

# ============================================
# 9. SIMPLE TRANSLATION FUNCTION
# ============================================
def translate(input_text):
    seq = eng_tokenizer.texts_to_sequences([input_text])
    seq = pad_sequences(seq, maxlen=max_len_eng, padding='post')
    pred = model.predict([seq, np.zeros((1, max_len_hin))], verbose=0)
    output = ""
    for i in np.argmax(pred[0], axis=1):
        for word, index in hin_tokenizer.word_index.items():
            if index == i:
                output += word + " "
    return output.strip()

# ============================================
# 10. TEST TRANSLATION
# ============================================
print("Input: hello")
print("Translated:", translate("hello"))

# ============================================================
# Expected Output:
#
# Model training complete.
# Input: hello
# Translated: namaste
#
# Note: Due to the very small dataset and augmentation, actual
# output may vary. The model demonstrates the encoder-decoder
# architecture for low-resource machine translation.
# ============================================================
