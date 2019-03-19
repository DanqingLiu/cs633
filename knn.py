import csv
import random
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial

def loadCsv(filename):
  '''
  load data. 
  https://stackoverflow.com/questions/4315506/load-csv-into-2d-matrix-with-numpy-for-plotting
  https://machinelearningmastery.com/index-slice-reshape-numpy-arrays-machine-learning-python/
  '''
  lines = csv.reader(open(filename, 'rt', encoding = 'utf-8'))
  next(lines, None)  # skip the headers
  dataset = list(lines)
  result = np.array(dataset).astype("float")
  np.random.shuffle(result) # randomly re-arrange the rows of the data samples
  X = result[:, 0:12]
  mean_X = np.mean(X, axis = 0)  # normalize the features 
  X -= mean_X
  y = result[:, -1]
  y[y>0] = 1
  return [X, y]

trainfile = 'train.csv'
[X_train, y_train]= loadCsv(trainfile)

testfile = 'test.csv'
[X_test, y_test]= loadCsv(testfile)

    
def train(X, y):
  X_train = X
  y_train = y

def compute_distances(X_test): # l2 norm, eucleadian distance
  num_test = X_test.shape[0]
  num_train = X_train.shape[0]
  dists = np.zeros((num_test, num_train)) 
  dists = (np.sum(X_test**2,axis=1)[:,np.newaxis] -2 * np.dot(X_test,X_train.T) + np.sum(X_train**2,axis=1))**0.5
  return dists

def compute_distances_v2(X_test): # hamming distance
  num_test = X_test.shape[0]
  num_train = X_train.shape[0]
  dists = np.zeros((num_test, num_train)) 
  for i in range(num_test):
    for j in range(num_train):
      dists[i][j] = scipy.spatial.distance.hamming(X_test[i], X_train[j])
  return dists

def predict_labels(dists, k=1):
  num_test = dists.shape[0]
  y_pred = np.zeros(num_test)
  for i in range(num_test):
    closest_y = []
    sort_dists_row = np.argsort(dists[i])
    k_nearest = sort_dists_row[0:k]        
    for j in range(k):
      closest_y.append(y_train[k_nearest[j]])

    label_num = {}
    most_common = 0
    for l in range(k):
      if closest_y[l] in label_num:
        label_num[closest_y[l]] = label_num[closest_y[l]] + 1
      else:
        label_num[closest_y[l]] = 1
    for key in label_num:
      if (label_num[key] > most_common):
        most_common = label_num[key]
        y_pred[i] = key           
  return y_pred

train(X_train, y_train)
#dists = compute_distances(X_test)
dists = compute_distances_v2(X_test)
y_test_pred = predict_labels(dists, k=8)

num_correct = np.sum(y_test_pred == y_test)
num_test = dists.shape[0]
accuracy = float(num_correct) / num_test
print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))

'''
Cross validation
'''
num_folds = 5
k_choices = [3, 5, 8, 10, 12, 15, 20, 50, 100]

X_train_folds = []
y_train_folds = []
X_train_folds = np.array_split(X_train, 5)
y_train_folds = np.array_split(y_train, 5)

k_to_accuracies = {}
for k in k_choices:
  k_to_accuracies[k] = []
  for fold in range(num_folds):
    includes = [x for x in range(num_folds) if x is not fold]
    ls_X_train = []
    ls_y_train = []
    for i in includes:
      ls_X_train.append(X_train_folds[i])
      ls_y_train.append(y_train_folds[i])

    X_train_v = np.concatenate(ls_X_train, axis=0)
    y_train_v = np.concatenate(ls_y_train, axis=0)
    X_test_v = X_train_folds[fold]
    y_test_v = y_train_folds[fold]

    train(X_train_v, y_train_v)
    dists = compute_distances(X_test_v)
    y_valid_pred = predict_labels(dists, k)
    num_correct = np.sum(y_valid_pred == y_test_v)
    num_valid = len(y_test_v)
    accuracy = float(num_correct) / num_valid
    k_to_accuracies[k].append(accuracy)

for k in k_choices:
  accuracies = k_to_accuracies[k]
  plt.scatter([k] * len(accuracies), accuracies)

accuracies_mean = np.array([np.mean(v) for k,v in sorted(k_to_accuracies.items())])
accuracies_std = np.array([np.std(v) for k,v in sorted(k_to_accuracies.items())])
plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std)
plt.title('Cross-validation on k')
plt.xlabel('k')
plt.ylabel('Cross-validation accuracy')
plt.show()
