import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy

iris_path = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'] 

df = pd.read_csv(iris_path, names = iris_names) 

X = df.iloc[:, [0, 1]].values
num_row = X.shape[0]
k = 3
centroid = X[np.random.choice(num_row, k, replace=False)]

def dist(x, y):
  return np.linalg.norm(x-y)

distance = np.zeros(k)
cluster_id = np.zeros(num_row)
centroid_prev = np.zeros(centroid.shape)
error = sum(dist(centroid[i], centroid_prev[i]) for i in range(k))
mean = np.zeros(centroid.shape)

iteration = 0
while error!=0:
  for i in range(num_row):
    for j in range(k):
      distance[j]= dist(X[i], centroid[j])
    cluster[i] = np.argmin(distance)
  centroid_prev = copy.deepcopy(centroid)
  for i in range(k):
    ele = [X[j] for j in range(num_row) if cluster[j] == i]
    centroid[i] = sum(ele)/len(ele)
  error = sum(dist(centroid[i], centroid_prev[i]) for i in range(k))

plt.scatter(X[:,0],X[:,1],c='black',label='unclustered data')
plt.xlabel('Income')
plt.ylabel('Number of transactions')
plt.legend()
plt.title('Plot of data points')
plt.show()

color=['red','blue','green']
labels=['cluster1','cluster2','cluster3']
clust = []
for i in range(k):
  ele = [X[j] for j in range(num_row) if cluster[j] == i]
  clust.append(np.array(ele))
for i in range(k):
    plt.scatter(clust[i][:,0],clust[i][:,1],c=color[i],label=labels[i])
plt.scatter(centroid[:,0],centroid[:,1],s=60,c='yellow',label='Centroids')
plt.xlabel('Income')
plt.ylabel('Number of transactions')
plt.legend()
plt.show()
