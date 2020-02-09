from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.externals.six import StringIO
import pydotplus
import numpy as np
import random
from numpy.random import RandomState
from scipy import stats
import math

#load the iris dataset
iris = load_iris()


#splitting dataset in training and test sets
random.seed(1118)
size_tr=50
n2=3
f = list(range(0,100))
g = list(range(0,50))
random.shuffle(f)
random.shuffle(g)
training_data=[]
training_class=[]
test_data= []
test_class= []
for i in range(0,size_tr-n2):
    training_data.append(iris.data[f[i]])
    training_class.append(iris.target[f[i]])    
for i in range(0,n2):
    training_data.append(iris.data[100+g[i]])
    training_class.append(iris.target[100+g[i]])    
for i in range(size_tr-n2,100):
    test_data.append(iris.data[f[i]])
    test_class.append(iris.target[f[i]])    
for i in range(n2,50):
    test_data.append(iris.data[100+g[i]])
    test_class.append(iris.target[100+g[i]])    
training_data=np.array(training_data)
training_class=np.array(training_class)
test_data=np.array(test_data)
test_class=np.array(test_class)


#building the classifier (the option random_state=RandomState(130) makes the algorithm deterministic)
clf = tree.DecisionTreeClassifier(criterion='gini', random_state=RandomState(130))
clf = clf.fit(training_data, training_class)

#print the decision tree in a pdf file
from sklearn.externals.six import StringIO

dot_data = StringIO() 
tree.export_graphviz(clf, out_file=dot_data) 
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("iris.pdf")

# the following code evaluates the decision tree on the test set and compute a confidence interval for
# the accuracy. You should create a list a, where a[i]=1 if the ith record test_data[i] has been classified
# correctly and 0 otherwise. Remember, a.append(1) add one more element to the list with value = 1.
a = []
pre_class = clf.predict(test_data)
for i in range(0, len(test_data)):
    if test_class[i] == pre_class[i]:
        a.append(1)
    else:
        a.append(0)
# fill properly this missing part


# The following code computes a confidence interval for the accuracy. The first argument is the confidence,
# e.g. 0.9, while the second argument of stats.norm.interval is the mean of the list a. Fill properly those
# parts.
CI = stats.norm.interval(0.9,float(sum(a))/max(len(a),1), scale = stats.sem(a))