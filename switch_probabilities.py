
import random
import math
from collections import defaultdict

class Node():

    def __init__(self, nodeid=None, left=None, right=None, leaf=None, prob=None):
        self.nodeid = nodeid
        self.leaf = leaf
        self.left = left
        self.right = right
        self.brlen = 100
        self.prob = prob

    def __repr__(self):
        if self.leaf is not None:
            return f"{{{self.nodeid}}}[{self.leaf}]"
        else:
            return f"({self.left}:{self.left.brlen}, {self.right}:{self.right.brlen}){{{self.nodeid}}}"

# def build_node_tree(tree):
#     if type(tree) is not tuple:
#         return Node(leaf=tree, prob=[int(tree==i) for i in range(2)])
#     return Node(left=build_node_tree(tree[0]), right=build_node_tree(tree[1]))

def build_node_tree(tree):
    """Builds an object tree from a list of tuples.

    Args:
        tree (tuple): Nested tuples representing a tree.

    Returns:
        Node: Root node of constructed tree.
        list: List of nodes in constructed tree.
    """
    def _build_node_tree(tree, nodelist):
        if type(tree) is not tuple:
            node = Node(leaf=tree, prob=[int(tree==i) for i in range(2)])
            nodelist.append(node)
            node.nodeid = len(nodelist)
            return node
        else:
            node = Node(left=_build_node_tree(tree[0], nodelist), right=_build_node_tree(tree[1], nodelist))
            nodelist.append(node)
            node.nodeid = len(nodelist)
            return node

    def _add_node_heights(node):
        if node.leaf is not None:
            node.height = 0
        else:
            _add_node_heights(node.left)
            _add_node_heights(node.right)
            node.height = node.left.height + node.left.brlen
            # assert node.left.height + node.left.brlen == node.right.height + node.right.brlen

    nodelist=[]
    nodetree =_build_node_tree(tree, nodelist)
    _add_node_heights(nodetree)

    return nodetree, nodelist


def mig_prob(a, b, t, m):
    """Computes the joint probability of label a and b separated by time t if rate of change is m.

    Args:
        a (str): Label of a
        b (str): Label of b
        t (float): Time between a and b
        mig_rate (float): Rate of change between a and b.

    Returns:
        float: Joint probability of label a and b
    """
    if a == b:
        return 1/2 + 1/2*math.exp(-t*m)
    else:
        return 1/2 - 1/2*math.exp(-t*m) 


def clade_likelihood(left_like_list, right_like_list, brlen_left, brlen_right, m):
    """Computes the likelihoods of the clade rooted by state 0 and 1.

    Args:
        left_like_list (list): Likelihoods of left child clade.
        right_like_list (list): Likelihoods of right child clade.
        brlen_left (float): Branch length to left child clade.
        brlen_right (float): Branch length to right child clade.
        m (float): Rate of change (migration rate) between population labels.

    Returns:
        list: Likelihoods of the clade rooted by state 0 and 1.
    """
    like_list = []
    for x in [0, 1]:
        like = 0
        for a, like_a in enumerate(left_like_list):
            for b, like_b in enumerate(right_like_list):
                like += mig_prob(x, a, brlen_left, m) * like_a * mig_prob(x, b, brlen_right, m) * like_b
        like_list.append(like)
    return like_list

def switch_sets(node, switch_rate):
    """Computes the likelihoods of the tree rooted by state 0 and 1.

    Args:
        node (Node): Root node.
        switch_rate (float): rate of switching between labels.

    Returns:
        list: Likelihoods of the tree rooted by state 0 and 1.
        list: Tuples of ids of node whoes parent branch has switch.
    """
    def fels(node, switch_rate, cargo):
        if node.leaf is not None:
            like = [int(x==node.leaf) for x in [0, 1]]
            return like, cargo
        
        # LEFT AND RIGHT WOULD BE POPED OFF STACK...
        left, _ = fels(node.left, switch_rate, cargo)
        right, _ = fels(node.right, switch_rate, cargo)
        brlen_left, brlen_right = node.left.brlen, node.right.brlen

        noswitch_like_list = clade_likelihood(left, right, brlen_left, brlen_right, switch_rate)
        if node.nodeid == root_node_id:
            like_list = noswitch_like_list
        else:
            like_list_left_switch = clade_likelihood(left[::-1], right, brlen_left, brlen_right, switch_rate)
            like_list_right_switch = clade_likelihood(left, right[::-1], brlen_left, brlen_right, switch_rate)

            weights = [sum(noswitch_like_list), sum(like_list_left_switch), sum(like_list_right_switch)] 
            i = random.choices([0, 1, 2], weights=weights, k = 1)[0]
            like_list = [noswitch_like_list, like_list_left_switch, like_list_right_switch][i]

            if i == 1:
                cargo.append(node.left.nodeid)
            elif i == 2:
                cargo.append(node.right.nodeid)

        return like_list, cargo

    root_node_id = node.nodeid

    return fels(node, switch_rate, [])


def sample_switch_sets(node_list, n_samples, mig_rate):
    """Samples switch sets and computes the relative probability of
       each set as well as the time interval for each switch.

    Args:
        node_list (list): List of tree nodes.
        n_samples (int): Number of sampled switch sets.
        mig_rate (float): Rate of switching between labels (migration rate).

    Returns:
        list: List of tuples each with switch sets, time intervals and relative probability.
    """
    node_tree = node_list[-1]

    counts = defaultdict(float)
    for i in range(n_samples):
        lst, cargo = switch_sets(node_tree, mig_rate)
        counts[tuple(cargo)] += sum(lst) # * 0.5**len(cargo)

    tot = sum(counts.values())
    for key in counts:
        counts[key] /= tot

    result = []
    for key, val in sorted([(v, k) for k, v in counts.items()], reverse=True):
        interv = [(node_list[n].height, node_list[n].height + node_list[n].brlen) for n in val]
        result.append((val, interv, key))

    return result


def cut_intervals(intervals, windowsize):
    intervals = intervals[::-1]
    trunc = windowsize
    buffer = []
    result = []
    while intervals:
        i = intervals.pop()

        if i[0] >= trunc:
            buffer.append(i)
            intervals.extend(buffer)
            buffer = []
            trunc += windowsize
        elif i[1] > trunc:
            c = i[:]
            c[1] = trunc
            result.append(c)
            c = i[:]
            c[0] = trunc            
            buffer.append(c)
        else:
            result.append(i)

        if not intervals and buffer:
            intervals.extend(buffer)
            buffer = []
            trunc += windowsize 

    return sorted(result)


assert 0

tree = ((((1, 1), 0), (1, 1)), 1)

node_tree, node_list = build_node_tree(tree)
print(node_tree)
print(node_list[-1])
print([x.nodeid for x in node_list])

samples = sample_switch_sets(node_list, n_samples=10000, mig_rate=0.00001)
for val, interv, key in samples:
    print(val, interv, key)

intervals = []
for val, interv, key in samples:
    for start, end in interv:
        intervals.append([start, end, key])
intervals = sorted(intervals)

# intervals = [[1, 5, 'A'], [1, 6, 'B'], [2, 5, 'C']]
# print(intervals)
# intervals = list(reversed(intervals))

binned_intervals = cut_intervals(intervals, windowsize=100)
bins = defaultdict(float)
for start, end, prob in binned_intervals:
    bins[(start, end)] += prob

for key in sorted(bins.keys()):
    print(key, bins[key])






# 1. POSTORDER COMPUTE COLOR PROBABILITY OF EACH INNER NODE
# 2. SAMPLE TREES:
    # 1. RANDOMLY SWAP EACH BRANCH AND RECORD SWAP
    # 2. WEIGHT SWAPS BY THE RESULTING LIKELIHOOD OF TREE


