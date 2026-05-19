# ============================================================
# Program 7:
# Write a Python program to find synonyms and antonyms of the
# word "active" using WordNet.
# ============================================================

# ============================================
# 1. IMPORT LIBRARY & DOWNLOAD WORDNET
# ============================================
import nltk

nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.corpus import wordnet

# ============================================
# 2. FIND SYNONYMS AND ANTONYMS
# ============================================
word = "active"
synonyms = []
antonyms = []

for syn in wordnet.synsets(word):
    for lemma in syn.lemmas():
        synonyms.append(lemma.name())
        if lemma.antonyms():
            antonyms.append(lemma.antonyms()[0].name())

# ============================================
# 3. REMOVE DUPLICATES
# ============================================
synonyms = set(synonyms)
antonyms = set(antonyms)

# ============================================
# 4. DISPLAY RESULTS
# ============================================
print("Word:", word)
print("\nSynonyms of 'active':")
print(synonyms)
print("\nAntonyms of 'active':")
print(antonyms)

# ============================================================
# Expected Output:
#
# Word: active
#
# Synonyms of 'active':
# {'active', 'active_voice', 'alive', 'participating', 'combat-ready',
#  'active_agent', 'dynamic', 'fighting'}
#
# Antonyms of 'active':
# {'passive', 'passive_voice', 'inactive', 'dormant', 'extinct',
#  'stative', 'quiet'}
# ============================================================
