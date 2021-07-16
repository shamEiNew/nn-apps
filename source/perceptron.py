#PERCEPTRON ALGORITHM

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Seed for reproducablitiy


def stepfunction(t):
  if t>=0:
    return 1
  return 0

def prediction(X, W, b):
  return stepfunction((np.dot(X, W.T)+b)[0])

def perceptronStep(X, y, W, b, alpha):

  for i in range(len(X)):
    y_pred = prediction(X[i], W, b)
    
    if y[i]-y_pred == -1:
       W= W - alpha*X[i]
       b = b - alpha
    
    elif y[i]-y_pred == 1:
      W = W + alpha*X[i]
      b = b + alpha
  return W, b

def trainPerceptronAlgorithm(X, y, alpha=0.01, num_epochs=25):
  np.random.seed(42)
  x_min, x_max = min(X.T[0]), max(X.T[0])
  y_min, y_max = min(X.T[1]), max(X.T[1])
  W = np.random.rand(1, 2)
  b = np.random.rand(1)[0]+x_max

  boundary_lines = []
  for i in range(num_epochs):
    W, b = perceptronStep(X, y, W, b, alpha)
    boundary_lines.append((-W[0][0]/W[0][1], -b/W[0][1]))
  return boundary_lines

def prepare(filename):
  with open(filename) as dataf:
    data = pd.read_csv(dataf, names=['x','y', 'label'])
  data = data.to_numpy()
  X = data[:, :2]
  y = data[:,2]
  return X, y


 # Plotting in 2 dimensional features.

def plot_points(X, y):
  admitted = X[np.argwhere(y==1)]
  rejected = X[np.argwhere(y==0)]
  fig = plt.figure()
  plt.scatter([s[0][0] for s in rejected], [s[0][1] for s in rejected], label ='0',s=25, color='red',edgecolor='k')
  plt.scatter([s[0][0] for s in admitted], [s[0][1] for s in admitted],label='1', s=25, color='blue',edgecolor='k')
  plt.legend()
  return fig


def display(fig_lines, m, b, color='g--', x_min=None, y_min=None,label=None):
  fig = fig_lines
  if x_min != None:
    plt.xlim(x_min[0], x_min[1])
    plt.ylim(y_min[0], y_min[1])
  x = np.arange(-10, 10, 0.01)
  if label:
    plt.plot(x, m*x+b, color, label=label)
  else:
    plt.plot(x, m*x+b, color)
  plt.legend()
  return fig