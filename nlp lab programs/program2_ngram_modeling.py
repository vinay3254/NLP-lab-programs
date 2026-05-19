# ============================================================
# Program 2:
# Demonstrate the N-gram modeling to analyze and establish the
# probability distribution across sentences and explore the
# utilization of unigrams, bigrams, and trigrams in diverse
# English sentences to illustrate the impact of varying n-gram
# orders on the calculated probabilities.
# ============================================================

# ============================================
# 1. IMPORT LIBRARIES
# ============================================
import nltk
from nltk.util import ngrams
from collections import Counter

nltk.download('punkt')
nltk.download('punkt_tab')

# ============================================
# 2. SAMPLE CORPUS
# ============================================
corpus = [
    "I love natural language processing",
    "I love machine learning",
    "natural language processing is fun",
    "machine learning is powerful"
]

# ============================================
# 3. TOKENIZATION
# ============================================
tokens = []
for sentence in corpus:
    words = nltk.word_tokenize(sentence.lower())
    tokens.extend(words)

# ============================================
# 4. UNIGRAM MODEL
# ============================================
unigrams = list(ngrams(tokens, 1))
unigram_freq = Counter(unigrams)
total_unigrams = sum(unigram_freq.values())

def unigram_prob(word):
    return unigram_freq[(word,)] / total_unigrams

# ============================================
# 5. BIGRAM MODEL
# ============================================
bigrams = list(ngrams(tokens, 2))
bigram_freq = Counter(bigrams)

def bigram_prob(w1, w2):
    return bigram_freq[(w1, w2)] / unigram_freq[(w1,)]

# ============================================
# 6. TRIGRAM MODEL
# ============================================
trigrams = list(ngrams(tokens, 3))
trigram_freq = Counter(trigrams)

def trigram_prob(w1, w2, w3):
    return trigram_freq[(w1, w2, w3)] / bigram_freq[(w1, w2)]

# ============================================
# 7. SENTENCE PROBABILITY FUNCTIONS
# ============================================
def sentence_prob_unigram(sentence):
    words = nltk.word_tokenize(sentence.lower())
    prob = 1
    for w in words:
        prob *= unigram_prob(w) if (w,) in unigram_freq else 1e-6
    return prob

def sentence_prob_bigram(sentence):
    words = nltk.word_tokenize(sentence.lower())
    prob = 1
    for w1, w2 in ngrams(words, 2):
        if (w1, w2) in bigram_freq:
            prob *= bigram_prob(w1, w2)
        else:
            prob *= 1e-6
    return prob

def sentence_prob_trigram(sentence):
    words = nltk.word_tokenize(sentence.lower())
    prob = 1
    for w1, w2, w3 in ngrams(words, 3):
        if (w1, w2, w3) in trigram_freq:
            prob *= trigram_prob(w1, w2, w3)
        else:
            prob *= 1e-6
    return prob

# ============================================
# 8. TEST SENTENCES
# ============================================
test_sentences = [
    "I love machine learning",
    "natural language processing is powerful",
    "I enjoy deep learning"
]

# ============================================
# 9. DISPLAY RESULTS
# ============================================
for sent in test_sentences:
    print("\n====================================")
    print("Sentence:", sent)
    print("Unigram Probability:", sentence_prob_unigram(sent))
    print("Bigram Probability:", sentence_prob_bigram(sent))
    print("Trigram Probability:", sentence_prob_trigram(sent))

# ============================================================
# Expected Output:
#
# ====================================
# Sentence: I love machine learning
# Unigram Probability: 0.00015241579027587256
# Bigram Probability: 0.5
# Trigram Probability: 0.5
#
# ====================================
# Sentence: natural language processing is powerful
# Unigram Probability: 8.467543904215141e-06
# Bigram Probability: 0.25
# Trigram Probability: 5e-07
#
# ====================================
# Sentence: I enjoy deep learning
# Unigram Probability: 1.2345679012345676e-14
# Bigram Probability: 9.999999999999999e-19
# Trigram Probability: 1e-12
# ============================================================
