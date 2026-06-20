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
        left = self.left_child.count_nodes_below(only_leaves) \
            if self.left_child else 0
        right = self.right_child.count_nodes_below(only_leaves) \
            if self.right_child else 0
        if only_leaves:
            return left + right
        return 1 + left + right

    def left_child_add_prefix(self, text):
        """Adds formatting prefix for left children."""
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("    |  " + x) + "\n"
        return new_text.rstrip("\n")

    def right_child_add_prefix(self, text):
        """Adds formatting prefix for right children."""
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("       " + x) + "\n"
        return new_text.rstrip("\n")

    def __str__(self):
        """String representation of the Node."""
        if self.is_root:
            res = (f"root [feature={self.feature}, "
                   f"threshold={self.threshold}]")
        else:
            res = (f"-> node [feature={self.feature}, "
                   f"threshold={self.threshold}]")

        if self.left_child:
            res += "\n" + self.left_child_add_prefix(str(self.left_child))
        if self.right_child:
            res += "\n" + self.right_child_add_prefix(str(self.right_child))
        return res

    def get_leaves_below(self):
        """Returns a list of all leaves below this node."""
        left = self.left_child.get_leaves_below() if self.left_child else []
        right = self.right_child.get_leaves_below() if self.right_child else []
        return left + right

    def update_bounds_below(self):
        """Recursively computes bounds for each node."""
        if self.is_root:
            self.upper = {0: np.inf}
            self.lower = {0: -1 * np.inf}

        for child in [self.left_child, self.right_child]:
            if child:
                child.lower = self.lower.copy()
                child.upper = self.upper.copy()

        # If value > threshold, goes left (threshold is lower bound)
        if self.left_child:
            self.left_child.lower[self.feature] = max(
                self.lower.get(self.feature, -np.inf), self.threshold)

        # If value <= threshold, goes right (threshold is upper bound)
        if self.right_child:
            self.right_child.upper[self.feature] = min(
                self.upper.get(self.feature, np.inf), self.threshold)

        for child in [self.left_child, self.right_child]:
            if child:
                child.update_bounds_below()


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
        return f"-> leaf [value={self.value}]"

    def get_leaves_below(self):
        """Returns the leaf itself as a list."""
        return [self]

    def update_bounds_below(self):
        """Pass since leaves do not have children to bound."""
        pass


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

    def get_leaves(self):
        """Returns all leaves in the tree."""
        return self.root.get_leaves_below()

    def update_bounds(self):
        """Updates all bounds in the tree."""
        self.root.update_bounds_below()

    def __str__(self):
        """String representation of the entire Tree."""
        return self.root.__str__()
