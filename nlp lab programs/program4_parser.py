# ============================================================
# Program 4:
# Write a program to implement top-down and bottom-up parser
# using appropriate context free grammar.
#
# Grammar: S -> a S b | ab
# Input:   aabb
# ============================================================

# ============================================
# 1. DEFINE TREE NODE
# ============================================
class Node:
    def __init__(self, value):
        self.value = value
        self.children = []

def print_tree(node, level=0):
    print(" " * level * 4 + node.value)
    for child in node.children:
        print_tree(child, level + 1)

input_string = "aabb"

# ============================================
# 2. TOP-DOWN PARSER (RETURNS TREE)
# ============================================
def parse_S(index):
    # Rule 1: S -> a S b
    if index < len(input_string) and input_string[index] == 'a':
        a_node = Node('a')
        result, child_S, next_index = parse_S(index + 1)
        if result and next_index < len(input_string) and input_string[next_index] == 'b':
            b_node = Node('b')
            root = Node('S')
            root.children = [a_node, child_S, b_node]
            return True, root, next_index + 1

    # Rule 2: S -> ab
    if (index + 1 < len(input_string) and
            input_string[index] == 'a' and
            input_string[index + 1] == 'b'):
        root = Node('S')
        root.children = [Node('a'), Node('b')]
        return True, root, index + 2

    return False, None, index

result_td, tree_td, final_index = parse_S(0)
if result_td and final_index == len(input_string):
    print("Top-Down Parse Tree:")
    print_tree(tree_td)
else:
    print("Top-Down Parsing Failed")

# ============================================
# 3. BOTTOM-UP PARSER (SHIFT-REDUCE)
# ============================================
stack = []
i = 0

def reduce_stack():
    global stack
    changed = True
    while changed:
        changed = False

        # Rule: S -> ab
        if len(stack) >= 2 and stack[-2][0] == 'a' and stack[-1][0] == 'b':
            a_node = stack[-2][1]
            b_node = stack[-1][1]
            new_node = Node('S')
            new_node.children = [a_node, b_node]
            stack = stack[:-2]
            stack.append(('S', new_node))
            changed = True

        # Rule: S -> a S b
        elif (len(stack) >= 3 and
              stack[-3][0] == 'a' and
              stack[-2][0] == 'S' and
              stack[-1][0] == 'b'):
            a_node = stack[-3][1]
            S_node = stack[-2][1]
            b_node = stack[-1][1]
            new_node = Node('S')
            new_node.children = [a_node, S_node, b_node]
            stack = stack[:-3]
            stack.append(('S', new_node))
            changed = True

while i < len(input_string):
    char = input_string[i]
    stack.append((char, Node(char)))
    i += 1
    reduce_stack()

reduce_stack()

if len(stack) == 1 and stack[0][0] == 'S':
    print("\nBottom-Up Parse Tree:")
    print_tree(stack[0][1])
else:
    print("Bottom-Up Parsing Failed")

# ============================================================
# Expected Output:
#
# Top-Down Parse Tree:
# S
#     a
#     S
#         a
#         b
#     b
#
# Bottom-Up Parse Tree:
# S
#     a
#     S
#         a
#         b
#     b
# ============================================================
