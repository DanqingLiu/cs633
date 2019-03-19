import csv
import random
import numpy as np
import matplotlib.pyplot as plt

def loadCsv(filename):
  '''
  load data. 
  https://stackoverflow.com/questions/4315506/load-csv-into-2d-matrix-with-numpy-for-plotting
  https://machinelearningmastery.com/index-slice-reshape-numpy-arrays-machine-learning-python/
  https://stackoverflow.com/questions/23911875/select-certain-rows-condition-met-but-only-some-columns-in-python-numpy
  '''
  lines = csv.reader(open(filename, 'rt', encoding = 'utf-8'))
  next(lines, None)  # skip the headers
  dataset = list(lines)
  result = np.array(dataset).astype("float")
  reduce_result = result[result[:,-1] > 0]
  np.random.shuffle(reduce_result) # randomly re-arrange the rows of the data samples
  X = reduce_result[:, 0:12]
  mean_X = np.mean(X, axis = 0)  # normalize the features 
  X -= mean_X
  y = reduce_result[:, -1]
  return [X, y]

trainfile = 'train.csv'
[X_train, y_train]= loadCsv(trainfile)

testfile = 'test.csv'
[X_test, y_test]= loadCsv(testfile)

    
def plot_outcome(y):
  # https://matplotlib.org/1.2.1/examples/pylab_examples/histogram_demo.html
  # the histogram of the data
  #n, bins, patches = plt.hist(y, 10, density=1, log = False, facecolor='blue', alpha=0.75)
  n, bins, patches = plt.hist(y, 10, density=1, log = True, facecolor='blue', alpha=0.75)

  plt.xlabel('Outcome variable(Forest fire area)')
  #plt.ylabel('Probability')
  plt.ylabel('Probability(log)')
  plt.title(r'Histogram of the outcome variable.')
  plt.axis([0, 300, 0, 0.3])
  plt.grid(True)
  plt.show()

#plot_outcome(y_train)
#plot_outcome(y_test)

def train(X, y):
  num_train = X.shape[0]
  X = np.c_[np.ones(num_train),X]
  X_t = X.T
  inv_x_2 = np.linalg.inv(X_t.dot(X))
  weight_ols = (inv_x_2.dot(X_t)).dot(y)
  return weight_ols

weight = train(X_train, y_train)
X_test = np.c_[np.ones(X_test.shape[0]),X_test]
y_pred = X_test.dot(weight)
RSS = ((y_test - y_pred).T).dot(y_test - y_pred)
print('The RSS error of the OLS solution: ')
print(RSS)

correlation = np.cov(y_test, y_pred)
print('Correlation between y_test and y_pred:')
print(correlation)
