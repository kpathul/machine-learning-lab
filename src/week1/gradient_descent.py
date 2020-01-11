import numpy as np

def get_predictions(X, theta):
    return X @ theta.T

def get_cost(X,y,theta):
    pred = get_predictions(X, theta)
    diff_squares = np.power((pred-y),2)
    cost = np.sum(diff_squares)/(2 * len(X))
    return cost

def gradient_descent(X,y,theta,epoch,learn_rate):
    cost = np.zeros(epoch)
    for i in range(epoch):
        pred = get_predictions(X, theta)
        theta = theta - (learn_rate/len(X)) * np.sum(X * (pred - y), axis=0)
        cost[i] = get_cost(X, y, theta)
    
    return theta,cost
