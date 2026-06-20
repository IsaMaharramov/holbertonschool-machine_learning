#!/usr/bin/env python3
"""Module for Decision Tree components (Task 5)."""
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
        res = (f"{prefix} [feature={self.feature}, "
               f"threshold={self.threshold}]\n")
        if self.left_child:
            res += self.left_child_add_prefix(str(self.left_child))
        if self.right_child:
            res += self.right_child_add_prefix(str(self.right_child))
        return res.strip()

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

        # If value > threshold, it goes left. So threshold is LOWER bound.
        if self.left_child:
            self.left_child.lower[self.feature] = max(
                self.lower.get(self.feature, -np.inf), self.threshold)

        # If value <= threshold, it goes right. So threshold is UPPER bound.
        if self.right_child:
            self.right_child.upper[self.feature] = min(
                self.upper.get(self.feature, np.inf), self.threshold)

        for child in [self.left_child, self.right_child]:
            if child:
                child.update_bounds_below()

    def update_indicator(self):
        """Computes the boolean indicator function for the node."""
        def is_large_enough(x):
            """Checks if values are greater than lower bounds."""
            return np.all(np.array([
                np.greater(x[:, k], self.lower[k]) for k in self.lower.keys()
            ]), axis=0)

        def is_small_enough(x):
            """Checks if values are less than or equal to upper bounds."""
            return np.all(np.array([
                np.less_equal(x[:, k], self.upper[k])
                for k in self.upper.keys()
            ]), axis=0)

        self.indicator = lambda x: np.all(np.array([
            is_large_enough(x), is_small_enough(x)
        ]), axis=0)


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
        """Leaves do not compute bounds; passed down from parent."""
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

    def __str__(self):
        """String representation of the entire Tree."""
        return self.root.__str__()

    def get_leaves(self):
        """Returns all leaves in the tree."""
        return self.root.get_leaves_below()

    def update_bounds(self):
        """Updates all bounds in the tree."""
        self.root.update_bounds_below()
