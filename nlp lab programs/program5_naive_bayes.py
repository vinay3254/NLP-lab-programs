# ============================================================
# Program 5:
# Given the following short movie reviews, each labeled with a
# genre, either comedy or action:
#
#   - "fun, couple, love, love"         -> comedy
#   - "fast, furious, shoot"            -> action
#   - "couple, fly, fast, fun, fun"     -> comedy
#   - "furious, shoot, shoot, fun"      -> action
#   - "fly, fast, shoot, love"          -> action
#
# A new document D: "fast, couple, shoot, fly"
#
# Compute the most likely class for D. Assume a Naive Bayes
# classifier and use add-1 (Laplace) smoothing for likelihoods.
# ============================================================

# ============================================
# 1. IMPORT LIBRARIES
# ============================================
from collections import defaultdict
import math

# ============================================
# 2. TRAINING DATA (GIVEN REVIEWS)
# ============================================
documents = [
    ("fun couple love love", "comedy"),
    ("fast furious shoot", "action"),
    ("couple fly fast fun fun", "comedy"),
    ("furious shoot shoot fun", "action"),
    ("fly fast shoot love", "action")
]

# ============================================
# 3. PREPROCESSING
# ============================================
vocab = set()
word_counts = {"comedy": defaultdict(int), "action": defaultdict(int)}
class_counts = {"comedy": 0, "action": 0}
total_words = {"comedy": 0, "action": 0}

for text, label in documents:
    class_counts[label] += 1
    words = text.split()
    for word in words:
        vocab.add(word)
        word_counts[label][word] += 1
        total_words[label] += 1

vocab_size = len(vocab)

# ============================================
# 4. PRIOR PROBABILITIES
# ============================================
total_docs = len(documents)
prior_comedy = class_counts["comedy"] / total_docs
prior_action = class_counts["action"] / total_docs

# ============================================
# 5. CLASSIFY NEW DOCUMENT
# ============================================
test_doc = "fast couple shoot fly"
test_words = test_doc.split()

log_prob_comedy = math.log(prior_comedy)
log_prob_action = math.log(prior_action)

for word in test_words:
    # Likelihood for comedy with Add-1 smoothing
    count = word_counts["comedy"][word]
    prob = (count + 1) / (total_words["comedy"] + vocab_size)
    log_prob_comedy += math.log(prob)

    # Likelihood for action with Add-1 smoothing
    count = word_counts["action"][word]
    prob = (count + 1) / (total_words["action"] + vocab_size)
    log_prob_action += math.log(prob)

# ============================================
# 6. OUTPUT RESULT
# ============================================
print("Log Probability (Comedy):", log_prob_comedy)
print("Log Probability (Action):", log_prob_action)

if log_prob_comedy > log_prob_action:
    print("\nPredicted Class: COMEDY")
else:
    print("\nPredicted Class: ACTION")

# ============================================================
# Expected Output:
#
# Log Probability (Comedy): -9.52173897104528
# Log Probability (Action): -8.671115273688494
# Predicted Class: ACTION
# ============================================================
