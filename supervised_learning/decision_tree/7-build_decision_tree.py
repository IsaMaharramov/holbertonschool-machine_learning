#!/usr/bin/env python3
"""Module for Decision Tree components with training logic."""
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
        res = f"{prefix} [feature={self.feature}, threshold={self.threshold}]\n"
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

        if self.left_child:
            self.left_child.lower[self.feature] = max(
                self.lower.get(self.feature, -np.inf), self.threshold)
        if self.right_child:
            self.right_child.upper[self.feature] = min(
                self.upper.get(self.feature, np.inf), self.threshold)

        for child in [self.left_child, self.right_child]:
            if child:
                child.update_bounds_below()

    def update_indicator(self):
        """Computes the boolean indicator function for the node."""
        def is_large_enough(x):
            return np.all(np.array([
                np.greater(x[:, k], self.lower[k]) for k in self.lower.keys()
            ]), axis=0)

        def is_small_enough(x):
            return np.all(np.array([
                np.less_equal(x[:, k], self.upper[k])
                for k in self.upper.keys()
            ]), axis=0)

        self.indicator = lambda x: np.all(np.array([
            is_large_enough(x), is_small_enough(x)
        ]), axis=0)

    def pred(self, x):
        """Recursively routes an observation."""
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        else:
            return self.right_child.pred(x)


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

    def get_leaves_below(self):
        """Returns the leaf itself as a list."""
        return [self]

    def update_bounds_below(self):
        """Leaves do not compute bounds."""
        pass

    def pred(self, x):
        """Returns the leaf value."""
        return self.value


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

    def update_predict(self):
        """Computes the final prediction function using array operations."""
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()          
        self.predict = lambda A: np.sum(
            np.array([leaf.indicator(A) * leaf.value for leaf in leaves]),
            axis=0)

    def pred(self, x):
        """Fallback prediction method (single observation)."""
        return self.root.pred(x)

    def np_extrema(self, arr):
        """Returns the minimum and maximum of a NumPy array."""
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """Selects a random feature and threshold for splitting."""
        diff = 0
        while diff == 0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            f_min, f_max = self.np_extrema(
                self.explanatory[:, feature][node.sub_population])
            diff = f_max - f_min
        x = self.rng.uniform()
        threshold = (1 - x) * f_min + x * f_max
        return feature, threshold

    def get_leaf_child(self, node, sub_population):
        """Creates a leaf child node calculating the majority class."""
        targets = self.target[sub_population]
        vals, counts = np.unique(targets, return_counts=True)
        value = vals[np.argmax(counts)]
        leaf_child = Leaf(value)
        leaf_child.depth = node.depth + 1
        leaf_child.subpopulation = sub_population
        leaf_child.sub_population = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """Creates an internal child node."""
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def fit_node(self, node):
        """Recursively grows the decision tree using array masking."""
        node.feature, node.threshold = self.split_criterion(node)

        mask = self.explanatory[:, node.feature] > node.threshold
        left_population = np.logical_and(node.sub_population, mask)
        right_population = np.logical_and(node.sub_population, ~mask)

        is_left_leaf = (node.depth + 1 >= self.max_depth or
                        np.sum(left_population) < self.min_pop or
                        np.unique(self.target[left_population]).size <= 1)

        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        is_right_leaf = (node.depth + 1 >= self.max_depth or
                         np.sum(right_population) < self.min_pop or
                         np.unique(self.target[right_population]).size <= 1)

        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def fit(self, explanatory, target, verbose=0):
        """Fits the Decision Tree to the data."""
        if self.split_criterion == "random": 
            self.split_criterion = self.random_split_criterion
        else: 
            self.split_criterion = self.Gini_split_criterion
        
        self.explanatory = explanatory
        self.target = target
        self.root.sub_population = np.ones_like(self.target, dtype='bool')

        self.fit_node(self.root)
        self.update_predict()

        if verbose == 1:
            print(f"""  Training finished.
    - Depth                     : {self.depth()}
    - Number of nodes           : {self.count_nodes()}
    - Number of leaves          : {self.count_nodes(only_leaves=True)}
    - Accuracy on training data : {self.accuracy(self.explanatory, self.target)}""")

    def accuracy(self, test_explanatory, test_target):
        """Computes the accuracy of predictions against target labels."""
        return np.sum(np.equal(self.predict(test_explanatory),
                               test_target)) / test_target.size
