
import tskit
tree = tskit.Tree.generate_random_binary(10, random_seed=2)

#labels = np.arra([0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0])

print(tree.draw_text())

for u in tree.postorder():

    print(u, tree.left_child_array[u], tree.right_child_array[u], 
        tree.parent_array[u], tree.edge_array[u],
        tree.branch_length(u))



# for node in tree.postorder([tree.root]):
#     tree.postorder([u])

#     tree.edge_array
#     tree.left_child_array
#     tree.right_child_array

