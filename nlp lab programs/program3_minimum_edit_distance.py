# ============================================================
# Program 3:
# Investigate the Minimum Edit Distance (MED) algorithm and its
# application in string comparison. The goal is to understand
# how the algorithm efficiently computes the minimum number of
# edit operations required to transform one string into another.
#
#   - Test the algorithm on strings with different variations
#     (e.g., typos, substitutions, insertions, deletions)
#   - Evaluate its adaptability to different types of input
#     variations
# ============================================================

# ============================================
# 1. IMPORT LIBRARY
# ============================================
import numpy as np

# ============================================
# 2. MED FUNCTION (LEVENSHTEIN DISTANCE)
# ============================================
def min_edit_distance(str1, str2):
    m = len(str1)
    n = len(str2)
    dp = np.zeros((m + 1, n + 1), dtype=int)

    for i in range(m + 1):
        dp[i][0] = i

    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                cost = 0
            else:
                cost = 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,       # Deletion
                dp[i][j - 1] + 1,       # Insertion
                dp[i - 1][j - 1] + cost # Substitution
            )

    return dp[m][n], dp

# ============================================
# 3. TEST CASES (DIFFERENT VARIATIONS)
# ============================================
tests = [
    ("cat", "cut"),      # Substitution
    ("cat", "cats"),     # Insertion
    ("cats", "cat"),     # Deletion
    ("kitten", "sitting"), # Mixed operations
    ("book", "back")     # Multiple substitutions
]

# ============================================
# 4. RUN TESTS
# ============================================
for s1, s2 in tests:
    distance, table = min_edit_distance(s1, s2)
    print("\n===================================")
    print("String 1:", s1)
    print("String 2:", s2)
    print("Minimum Edit Distance:", distance)
    print("\nDP Table:")
    print(table)

# ============================================
# 5. SIMILARITY EVALUATION FUNCTION
# ============================================
def evaluate_similarity(distance, max_len):
    similarity = 1 - (distance / max_len)
    return round(similarity, 2)

print("\n===== SIMILARITY SCORES =====")
for s1, s2 in tests:
    distance, _ = min_edit_distance(s1, s2)
    max_len = max(len(s1), len(s2))
    score = evaluate_similarity(distance, max_len)
    print(f"{s1} -> {s2} : Similarity =", score)

# ============================================================
# Expected Output:
#
# ===================================
# String 1: cat
# String 2: cut
# Minimum Edit Distance: 1
# DP Table:
# [[0 1 2 3]
#  [1 0 1 2]
#  [2 1 1 2]
#  [3 2 2 1]]
#
# ===================================
# String 1: cat
# String 2: cats
# Minimum Edit Distance: 1
# DP Table:
# [[0 1 2 3 4]
#  [1 0 1 2 3]
#  [2 1 0 1 2]
#  [3 2 1 0 1]]
#
# ===================================
# String 1: cats
# String 2: cat
# Minimum Edit Distance: 1
# DP Table:
# [[0 1 2 3]
#  [1 0 1 2]
#  [2 1 0 1]
#  [3 2 1 0]
#  [4 3 2 1]]
#
# ===================================
# String 1: kitten
# String 2: sitting
# Minimum Edit Distance: 3
# DP Table:
# [[0 1 2 3 4 5 6 7]
#  [1 1 2 3 4 5 6 7]
#  [2 2 1 2 3 4 5 6]
#  [3 3 2 1 2 3 4 5]
#  [4 4 3 2 1 2 3 4]
#  [5 5 4 3 2 2 3 4]
#  [6 6 5 4 3 3 2 3]]
#
# ===================================
# String 1: book
# String 2: back
# Minimum Edit Distance: 2
# DP Table:
# [[0 1 2 3 4]
#  [1 0 1 2 3]
#  [2 1 1 2 3]
#  [3 2 2 2 3]
#  [4 3 3 3 2]]
#
# ===== SIMILARITY SCORES =====
# cat -> cut : Similarity = 0.67
# cat -> cats : Similarity = 0.75
# cats -> cat : Similarity = 0.75
# kitten -> sitting : Similarity = 0.57
# book -> back : Similarity = 0.5
# ============================================================
