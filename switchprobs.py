
import tskit
import numpy as np

import random
import math
from collections import defaultdict
from functools import partial



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

# def switch_sets(node, switch_rate):
#     """Computes the likelihoods of the tree rooted by state 0 and 1.

#     Args:
#         node (Node): Root node.
#         switch_rate (float): rate of switching between labels.

#     Returns:
#         list: Likelihoods of the tree rooted by state 0 and 1.
#         list: Tuples of ids of node whoes parent branch has switch.
#     """
#     def fels(node, switch_rate, cargo):
#         if node.leaf is not None:
#             like = [int(x==node.leaf) for x in [0, 1]]
#             return like, cargo
        
#         # LEFT AND RIGHT WOULD BE POPED OFF STACK...
#         left, _ = fels(node.left, switch_rate, cargo)
#         right, _ = fels(node.right, switch_rate, cargo)
#         brlen_left, brlen_right = node.left.brlen, node.right.brlen

#         noswitch_like_list = clade_likelihood(left, right, brlen_left, brlen_right, switch_rate)
#         if node.nodeid == root_node_id:
#             like_list = noswitch_like_list
#         else:
#             like_list_left_switch = clade_likelihood(left[::-1], right, brlen_left, brlen_right, switch_rate)
#             like_list_right_switch = clade_likelihood(left, right[::-1], brlen_left, brlen_right, switch_rate)

#             weights = [sum(noswitch_like_list), sum(like_list_left_switch), sum(like_list_right_switch)] 
#             i = random.choices([0, 1, 2], weights=weights, k = 1)[0]
#             like_list = [noswitch_like_list, like_list_left_switch, like_list_right_switch][i]

#             if i == 1:
#                 cargo.append(node.left.nodeid)
#             elif i == 2:
#                 cargo.append(node.right.nodeid)

#         return like_list, cargo

#     root_node_id = node.nodeid

#     return fels(node, switch_rate, [])




def sample_switch_sets(tree, labels, switch_rate, n_samples):
    """Samples switch sets and computes the relative probability of
       each set as well as the time interval for each switch.

    Args:
        node_list (list): List of tree nodes.
        n_samples (int): Number of sampled switch sets.
        mig_rate (float): Rate of switching between labels (migration rate).

    Returns:
        list: List of tuples each with switch sets, time intervals and relative probability.
    """

    part_fun = partial(fun, switch_rate=0.001, tree=tree)

    counts = defaultdict(float)
    for i in range(n_samples):
        lst, cargo = switch_sets(tree, labels, part_fun)
        # counts[tuple(cargo)] += sum(lst)
        # counts[tuple(cargo)] += 1 / sum(lst)
        # counts[tuple(cargo)] += sum(lst) / 0.5**len(cargo)
        counts[tuple(cargo)] += 1 / len(cargo)
        # counts[tuple(cargo)] += 1 

    tot = sum(counts.values())
    for key in counts:
        counts[key] /= tot

    result = []
    for key, val in sorted([(v, k) for k, v in counts.items()], reverse=True):
        event = []
        for n, l in val:
            start = tree.tree_sequence.node(n).time
            end = start + tree.branch_length(n)
            event.append((start, end, l))
        # event = [(node_list[n].height, node_list[n].height + node_list[n].brlen) for n in val]
        result.append((val, event, key))

    return result


def cut_intervals(intervals, windowsize):
    intervals = intervals[::-1]
    windowsize = float(windowsize)
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


def switch_sets(tree, labels, f):

    # stack for values computed in traversal
    stack = []
    cargo = []

    # iterate over nodes postorder
    for u in tree.postorder():
        if tree.is_leaf(u):
            like = [int(x==labels[u]) for x in [0, 1]]
            stack.append(like)
        else:
            # pop two node values (pop in this order to match right, left order in postorder)
            right = stack.pop()
            left = stack.pop()
            # push their parent value
            like, switch_node_index, label = f(u, left, right)  
            if switch_node_index is not None:
                cargo.append((switch_node_index, label))
            stack.append(like)
            
    return stack[0], cargo


def fun(u, left_like, right_like, switch_rate, tree):

    brlen_left = tree.branch_length(tree.left_child_array[u])
    brlen_right = tree.branch_length(tree.right_child_array[u])

    noswitch_like_list = clade_likelihood(left_like, right_like, brlen_left, brlen_right, switch_rate)
    like_list_left_switch = clade_likelihood(left_like[::-1], right_like, brlen_left, brlen_right, switch_rate)
    like_list_right_switch = clade_likelihood(left_like, right_like[::-1], brlen_left, brlen_right, switch_rate)

    weights = [sum(noswitch_like_list), sum(like_list_left_switch), sum(like_list_right_switch)] 
    i = random.choices([0, 1, 2], weights=weights, k = 1)[0]

    label = None
    if i:
        label = np.argmin([left_like, right_like][i-1])

    like_list = [noswitch_like_list, like_list_left_switch, like_list_right_switch][i]
    switch_node_idx = [None, tree.left_child_array[u],  tree.right_child_array[u]][i]

    return like_list, switch_node_idx, label

# # tree = tskit.Tree.generate_random_binary(9, random_seed=3)

# # tree = tskit.Tree.generate_balanced(9, arity=2)
# # labels = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0])

# n = 20
# tree = tskit.Tree.generate_balanced(n, arity=2)
# leaves = list(range(n))
# labels = np.array([0] * 10 + [1] * 10)
# labels[0] = 1
# # labels[2] = 1
# # labels[4] = 1

# # tree = tskit.Tree.generate_balanced(5, arity=2)
# # labels = np.array([1, 1, 0, 1, 1])

# print(tree.draw_text())

# print()

# # part_fun = partial(fun, switch_rate=0.001,
# #    left_child_array=tree.left_child_array,
# #    right_child_array=tree.right_child_array, 
# #    parent_array=tree.parent_array,
# #    num_children_array=tree.num_children_array)
# # part_fun = partial(fun, switch_rate=0.001, tree=tree)

# print(np.array(range(len(labels))))
# print(labels)
# # like, cargo = switch_sets(tree, labels, part_fun)

# samples = sample_switch_sets(tree, labels, n_samples=1000, switch_rate=0.001)
# for val, interv, key in samples:
#     print(val, interv, key)

# intervals = []
# for val, interv, key in samples:
#     for start, end, label in interv:
#         intervals.append([start, end, key, label])
# intervals = sorted(intervals)

# print("======")

# binned_intervals = cut_intervals(intervals, windowsize=1)
# bins = defaultdict(float)
# for start, end, prob, label in binned_intervals:
#     bins[(start, end, label)] += prob

# mrca_tree = tree.time(tree.root)
# mrca_0 = tree.time(tree.mrca(*[x for x in leaves if labels[x] == 0]))
# mrca_1 = tree.time(tree.mrca(*[x for x in leaves if labels[x] == 1]))

# for key in sorted(bins.keys()):
#     print(*key, round(bins[key], 5), mrca_tree, mrca_0, mrca_1, sep='\t')

# # print('tmrca', )
# # print('mrca of 0', tree.time(tree.mrca(*[x for x in leaves if labels[x] == 0])))
# # print('mrca of 1', tree.time(tree.mrca(*[x for x in leaves if labels[x] == 1])))



# #labels = np.arrau([0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0])

# # display(SVG(tree.draw_svg()))

# # for u in tree.postorder():

# #     print(u, tree.left_child_array[u], tree.right_child_array[u], tree.parent_array[u])



# # for node in tree.postorder([tree.root]):
# #     tree.postorder([u])

# #     tree.edge_array
# #     tree.left_child_array
# #     tree.right_child_array



# #Tree.time(u) Returns the time of the specified node.
# #Tree.is_leaf(u)  Returns True if the specified node is a leaf.
# #Tree.is_internal(u)  Returns True if the specified node is not a leaf.
# #Tree.branch_length(u)   Returns the length of the branch (in units of time) joining the specified node to its parent.
# # Tree.interval   Returns the coordinates of the genomic interval that this tree represents the history of.






#     # print(tree.edge_array)
#     # print(tree.postorder())
#     # print([tree.tree_sequence.node(i).time for i in tree.postorder()])
#     # print()
#     # print([tree.branch_length(i) for i in tree.postorder()])
#     # # print([tree.edge_array[i] for i in tree.postorder()])
#     # print()


#     # #    left_child_array, right_child_array, parent_array, num_children_array

