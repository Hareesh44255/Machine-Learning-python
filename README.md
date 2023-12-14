from sklearn.tree import export_graphviz
import graphviz

# ... (Previous code remains the same until decision tree creation)

# Convert the decision tree into a format suitable for visualization
def visualize_tree(tree):
    dot_data = export_graphviz(tree, out_file=None, feature_names=features, class_names=df['PlayTennis'].unique(), filled=True, rounded=True)
    graph = graphviz.Source(dot_data)
    graph.render('tennis_play_decision_tree')  # Save the tree visualization to a file
    return graph

# Visualize the decision tree
tree_graph = visualize_tree(decision_tree)
tree_graph.view()
