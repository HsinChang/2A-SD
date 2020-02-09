from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import numpy as np
import pandas as pd

#Processing the data
original_data = pd.DataFrame(pd.read_csv('dow_jones_index.data'))
data_needed = original_data[['stock','percent_change_price']]
S_name = []
a = []
for index, row in data_needed.iterrows():
    u = row["stock"]
    v = row["percent_change_price"]
    if not ( u in S_name):
        S_name.append(u)
        p = [v]
        a.append(p)
    else:
        a[S_name.index(u)].append(v)
Y = np.array(a, dtype='float32')
'''
#normalize
Y = normalize(Y, 'l2')
'''
#K-means algorithm
kmeans = KMeans(init='random', n_clusters=8, max_iter=10000, n_init=100).fit(Y)

#Calculate the SSE
labels = kmeans.labels_
centers = kmeans.cluster_centers_
sse = 0.0
for i in range(0, Y.shape[0] - 1):
    stock = Y[i]
    center_index = labels[i]
    center = centers[center_index]
    for j in range(0, Y.shape[1] - 1):
        dist = stock[j] - center[j]
        sse += np.square(dist)
print(sse)