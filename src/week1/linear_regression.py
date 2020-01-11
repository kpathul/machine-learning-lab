import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from gradient_descent import *

path = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat'
df = pd.read_csv(path, sep='\t',  header=None)

df.columns = ['frequency', 'angle-of-attack', 'chord-length', 'free-stream-velocity', 'suction-side-thickness', 'sound-pressure-level']

df = (df - df.mean())/df.std()
num_col = df.shape[1]

X= df.iloc[:, 0:num_col-1]
ones= np.ones([X.shape[0], 1])
X = np.concatenate((ones,X),axis=1)

y = df.iloc[:,num_col-1:].values

epoch= 100000
learn_rate= 0.001

theta = np.zeros([1,num_col])

new_theta,cost = gradient_descent(X,y,theta,epoch,learn_rate)
print(new_theta)

cost = get_cost(X,y,new_theta)
print(cost)
