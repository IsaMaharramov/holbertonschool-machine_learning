#!/usr/bin/env python3
"""Module for printing Decision Tree components."""
import numpy as np


class Node:
    """Class representing a Node in a decision tree."""
    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        """Initializes a Node."""
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """Calculates the maximum depth below this node."""
        left = self.left_child.max_depth_below() if self.left_child else 0
        right = self.right_child.max_depth_below() if self.right_child else 0
        return max(left, right)

    def count_nodes_below(self, only_leaves=False):
        """Counts the nodes below this node."""
        left = self.left_child.count_nodes_below(only_leaves)
        right = self.right_child.count_nodes_below(only_leaves)
        if only_leaves:
            return left + right
        return 1 + left + right

    def left_child_add_prefix(self, text):
        """Adds formatting prefix for left children."""
        lines = text.split("\n")
        new_text = "    +---> " + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("    |      " + x) + "\n"
        return new_text

    def right_child_add_prefix(self, text):
        """Adds formatting prefix for right children."""
        lines = text.split("\n")
        new_text = "    +---> " + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("           " + x) + "\n"
        return new_text

    def __str__(self):
        """String representation of the Node."""
        prefix = "root" if self.is_root else "node"
        res = f"{prefix} [feature={self.feature}, threshold={self.threshold}]\n"
        if self.left_child:
            res += self.left_child_add_prefix(str(self.left_child))
        if self.right_child:
            res += self.right_child_add_prefix(str(self.right_child))
        return res.strip()


class Leaf(Node):
    """Class representing a Leaf in a decision tree."""
    def __init__(self, value, depth=None):
        """Initializes a Leaf."""
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """Returns the depth of the leaf."""
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """Counts the leaf itself."""
        return 1

    def __str__(self):
        """String representation of the Leaf."""
        return (f"-> leaf [value={self.value}]")


class Decision_Tree():
    """Class representing a Decision Tree."""
    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        """Initializes a Decision Tree."""
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        """Returns the maximum depth of the tree."""
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """Counts total nodes in the tree."""
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """String representation of the entire Tree."""
        return self.root.__str__()
