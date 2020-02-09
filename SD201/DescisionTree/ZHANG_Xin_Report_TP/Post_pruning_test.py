from sklearn import tree
from sklearn.externals.six import StringIO
import pydotplus
import numpy as np

#Load the data and training
sky_data = np.loadtxt('skysurvey/training_data.csv', delimiter=",")
sky_class = np.loadtxt('skysurvey/training_class.csv')
clf = tree.DecisionTreeClassifier(min_samples_leaf=0.01)
clf.fit(sky_data, sky_class)

def post_pruning(tree):
    min_leaf = 1
    tempm = tree.min_samples_leaf
    if tempm != None:
        if tempm < 1:
            min_leaf = tempm * clf.tree_.n_node_samples[0]
        if tempm > 1:
            min_leaf = tempm
    array_children = tree.tree_.children_left
    num_leaves = np.count_nonzero(array_children == -1)
    n = tree.n_classes_
    if n >= num_leaves:
        raise Exception('Tree can no longer be pruned.')
    else:
        for i in range(tree.tree_.node_count):
            values = tree.tree_.value[i,0]
            instance_misplaced = sum(values)-max(values)
            if instance_misplaced < min_leaf:
                tree.tree_.children_left[i] = -1
                tree.tree_.children_right[i] = -1

post_pruning(clf)

dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data, filled= True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("Tree_pruned.pdf")
