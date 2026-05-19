# ============================================================
# Program 6:
# Demonstrate the following using appropriate programming tool
# which illustrates the use of information retrieval in NLP:
#
#   - Study the various Corpus: Brown, Inaugural, Reuters, UDHR
#     with methods like fileids, raw, words, sents, categories
#   - Create and use your own corpora (plaintext, categorical)
#   - Study Conditional Frequency Distributions
#   - Study of tagged corpora with tagged_sents, tagged_words
#   - Write a program to find the most frequent noun tags
#   - Map Words to Properties Using Python Dictionaries
#   - Study Rule-based tagger, Unigram Tagger
#   - Find different words from a given plain text without any
#     space by comparing with a corpus; also find the score.
# ============================================================

# ============================================
# 1. INSTALL & DOWNLOAD NLTK DATA
# ============================================
import nltk
import os

nltk.download('brown')
nltk.download('inaugural')
nltk.download('reuters')
nltk.download('udhr')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')

# ============================================
# 2. IMPORT LIBRARIES
# ============================================
from nltk.corpus import brown, inaugural, reuters, udhr
from nltk import ConditionalFreqDist, FreqDist
from nltk.corpus import PlaintextCorpusReader, CategorizedPlaintextCorpusReader

# ============================================
# 3. STUDY VARIOUS CORPORA
# ============================================
print("\n===== CORPORA STUDY =====")
print("\nBrown Categories:", brown.categories()[:5])
print("Brown Words:", brown.words()[:10])
print("Brown Sentences:", brown.sents()[:2])

print("\nInaugural Files:", inaugural.fileids()[:5])
print("Inaugural Raw Text:", inaugural.raw()[:200])

print("\nReuters Categories:", reuters.categories()[:5])
print("Reuters Words:", reuters.words()[:10])

print("\nUDHR Files:", udhr.fileids()[:5])
print("UDHR Words:", udhr.words()[:10])

# ============================================
# 4. CREATE YOUR OWN CORPUS
# ============================================
print("\n===== CUSTOM CORPUS =====")
os.makedirs('mycorpus/sports', exist_ok=True)
os.makedirs('mycorpus/news', exist_ok=True)

with open('mycorpus/sports/s1.txt', 'w') as f:
    f.write("Cricket is a popular sport in India. Players performed well.")

with open('mycorpus/news/n1.txt', 'w') as f:
    f.write("The government announced a new policy. Economy is improving.")

my_corpus = PlaintextCorpusReader('mycorpus', r'.*\.txt')
print("Files:", my_corpus.fileids())
print("Words:", my_corpus.words()[:10])

cat_corpus = CategorizedPlaintextCorpusReader(
    'mycorpus',
    r'.*\.txt',
    cat_pattern=r'(\w+)/.*'
)
print("Categories:", cat_corpus.categories())

# ============================================
# 5. CONDITIONAL FREQUENCY DISTRIBUTION
# ============================================
print("\n===== CONDITIONAL FREQUENCY =====")
cfd = ConditionalFreqDist(
    (genre, word)
    for genre in brown.categories()
    for word in brown.words(categories=genre)
)
print("Frequency of 'the' in news:", cfd['news']['the'])

# ============================================
# 6. TAGGED CORPUS
# ============================================
print("\n===== TAGGED CORPUS =====")
tagged_words = brown.tagged_words()[:10]
tagged_sents = brown.tagged_sents()[:2]
print("Tagged Words:", tagged_words)
print("Tagged Sentences:", tagged_sents)

# ============================================
# 7. MOST FREQUENT NOUN TAGS
# ============================================
print("\n===== MOST FREQUENT NOUN TAGS =====")
words_tags = brown.tagged_words()
noun_tags = [tag for (word, tag) in words_tags if tag.startswith('NN')]
fd = FreqDist(noun_tags)
print("Top Noun Tags:", fd.most_common(5))

# ============================================
# 8. WORD PROPERTY MAPPING (DICTIONARY)
# ============================================
print("\n===== WORD PROPERTIES =====")
word_properties = {
    "run": {"POS": "verb", "tense": "present"},
    "ran": {"POS": "verb", "tense": "past"},
    "dog": {"POS": "noun", "type": "animal"}
}
print("Properties of 'dog':", word_properties["dog"])

# ============================================
# 9. RULE-BASED TAGGER
# ============================================
print("\n===== RULE-BASED TAGGER =====")
patterns = [
    (r'.*ing$', 'VBG'),
    (r'.*ed$', 'VBD'),
    (r'.*s$', 'NNS'),
    (r'.*', 'NN')
]
rule_tagger = nltk.RegexpTagger(patterns)
print("Rule Tagger Output:", rule_tagger.tag(["playing", "played", "dogs"]))

# ============================================
# 10. UNIGRAM TAGGER
# ============================================
print("\n===== UNIGRAM TAGGER =====")
train_sents = brown.tagged_sents(categories='news')
unigram_tagger = nltk.UnigramTagger(train_sents)
print("Unigram Tagger Output:", unigram_tagger.tag(["The", "dog", "runs"]))

# ============================================
# 11. WORD SEGMENTATION (NO-SPACE TEXT)
# ============================================
print("\n===== WORD SEGMENTATION =====")
def word_segmentation(text, corpus_words):
    result = []
    i = 0
    while i < len(text):
        for j in range(len(text), i, -1):
            word = text[i:j]
            if word in corpus_words:
                result.append(word)
                i = j - 1
                break
        i += 1
    return result

corpus_words = set(brown.words())
text = "thisisatest"
segmented = word_segmentation(text, corpus_words)
print("Segmented Words:", segmented)
score = len(segmented)
print("Score:", score)

# ============================================================
# Expected Output:
#
# ===== CORPORA STUDY =====
# Brown Categories: ['adventure', 'belles_lettres', 'editorial', 'fiction', 'government']
# Brown Words: ['The', 'Fulton', 'County', 'Grand', 'Jury', 'said', 'Friday', 'an', 'investigation', 'of']
# ...
# ===== CUSTOM CORPUS =====
# Files: ['news/n1.txt', 'sports/s1.txt']
# Words: ['The', 'government', 'announced', 'a', 'new', 'policy', '.', 'Economy', 'is', 'improving']
# Categories: ['news', 'sports']
#
# ===== CONDITIONAL FREQUENCY =====
# Frequency of 'the' in news: 5580
#
# ===== TAGGED CORPUS =====
# Tagged Words: [('The', 'AT'), ('Fulton', 'NP-TL'), ...]
#
# ===== MOST FREQUENT NOUN TAGS =====
# Top Noun Tags: [('NN', 152470), ('NNS', 55110), ('NN-TL', 13372), ('NNS-TL', 2226), ('NN$', 1480)]
#
# ===== WORD PROPERTIES =====
# Properties of 'dog': {'POS': 'noun', 'type': 'animal'}
#
# ===== RULE-BASED TAGGER =====
# Rule Tagger Output: [('playing', 'VBG'), ('played', 'VBD'), ('dogs', 'NNS')]
#
# ===== UNIGRAM TAGGER =====
# Unigram Tagger Output: [('The', 'AT'), ('dog', 'NN'), ('runs', 'NNS')]
#
# ===== WORD SEGMENTATION =====
# Segmented Words: ['this', 'is', 'ate', 't']
# Score: 4
# ============================================================
