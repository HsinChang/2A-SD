from sklearn import tree
from sklearn.externals.six import StringIO
import pydotplus
import numpy as np

#Load the data and training
sky_data = np.loadtxt('skysurvey/training_data.csv', delimiter=",")
sky_class = np.loadtxt('skysurvey/training_class.csv')
clf = tree.DecisionTreeClassifier(min_samples_leaf=0.01)
clf.fit(sky_data, sky_class)

#Draw the tree into a pdf
dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data, filled= True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("Tree.pdf")

#Calculate the generalization error
training_error = (len(sky_class) - clf.score(sky_data,sky_class)*len(sky_class))/len(sky_class)
array_children = clf.tree_.children_left
num_leaves = np.count_nonzero(array_children == -1)
gen_error = training_error + (0.5*num_leaves/len(sky_class))
print("training errors:"+str(100*training_error)+"%")
print("Number of nodes in the tree:" +str(clf.tree_.node_count))
print('Generalization error: ' + str(100*gen_error) + '%')
