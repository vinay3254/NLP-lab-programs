# ============================================================
# Program 1:
# Write a Python program for the following preprocessing of
# text in NLP:
#   - Tokenization
#   - Filtration
#   - Script Validation
#   - Stop Word Removal
#   - Stemming
# ============================================================

# ============================================
# 1. IMPORT LIBRARIES
# ============================================
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

# ============================================
# 2. INPUT TEXT
# ============================================
text = "Hello! This is an NLP example. Running, runs, and ran are forms of run. 123 @#$"

# ============================================
# 3. TOKENIZATION
# ============================================
tokens = word_tokenize(text)
print("Tokens:", tokens)

# ============================================
# 4. FILTRATION (REMOVE PUNCTUATION & NUMBERS)
# ============================================
filtered_tokens = []
for token in tokens:
    if re.match("^[A-Za-z]+$", token):
        filtered_tokens.append(token)
print("\nFiltered Tokens:", filtered_tokens)

# ============================================
# 5. SCRIPT VALIDATION (CHECK ENGLISH / ASCII)
# ============================================
validated_tokens = []
for token in filtered_tokens:
    if token.isascii():
        validated_tokens.append(token.lower())
print("\nScript Validated Tokens:", validated_tokens)

# ============================================
# 6. STOP WORD REMOVAL
# ============================================
stop_words = set(stopwords.words('english'))
clean_tokens = []
for token in validated_tokens:
    if token not in stop_words:
        clean_tokens.append(token)
print("\nAfter Stopword Removal:", clean_tokens)

# ============================================
# 7. STEMMING
# ============================================
stemmer = PorterStemmer()
stemmed_tokens = []
for token in clean_tokens:
    stemmed_tokens.append(stemmer.stem(token))
print("\nStemmed Tokens:", stemmed_tokens)

# ============================================================
# Expected Output:
#
# Tokens: ['Hello', '!', 'This', 'is', 'an', 'NLP', 'example',
#          '.', 'Running', ',', 'runs', ',', 'and', 'ran', 'are',
#          'forms', 'of', 'run', '.', '123', '@', '#', '$']
#
# Filtered Tokens: ['Hello', 'This', 'is', 'an', 'NLP', 'example',
#                   'Running', 'runs', 'and', 'ran', 'are', 'forms',
#                   'of', 'run']
#
# Script Validated Tokens: ['hello', 'this', 'is', 'an', 'nlp',
#                            'example', 'running', 'runs', 'and',
#                            'ran', 'are', 'forms', 'of', 'run']
#
# After Stopword Removal: ['hello', 'nlp', 'example', 'running',
#                           'runs', 'ran', 'forms', 'run']
#
# Stemmed Tokens: ['hello', 'nlp', 'exampl', 'run', 'run', 'ran',
#                  'form', 'run']
# ============================================================
