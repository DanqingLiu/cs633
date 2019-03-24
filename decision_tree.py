import csv
import random
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial
import math

def load_csv(filename):
  '''
  load data.
  https://stackoverflow.com/questions/4315506/load-csv-into-2d-matrix-with-numpy-for-plotting
  https://machinelearningmastery.com/index-slice-reshape-numpy-arrays-machine-learning-python/
  '''
  lines = csv.reader(open(filename, 'rt', encoding = 'utf-8'))
  dataset = list(lines)
  result = np.array(dataset).astype("int")
  np.random.shuffle(result) # randomly re-arrange the rows of the data samples
  return result

def label_counts(data):
  '''
  https://www.w3resource.com/python-exercises/numpy/python-numpy-exercise-94.php
  https://stackoverflow.com/questions/52207358/create-dictionary-from-two-numpy-arrays
  '''
  counts = {}
  for d in data:
    label = d[-1]
    if label not in counts:
      counts[label] = 0
    counts[label] += 1
  return counts

#Decision tree introduction. https://www.youtube.com/watch?v=LDRbO9a6XPU

class Question:
  '''
  A Question is used to partition a dataset.
  '''
  def __init__(self, column, value):
      self.column = column
      self.value = value

  def match(self, example):
      # Compare the feature value in an example to the feature value in this question.
      val = example[self.column]
      return val >= self.value

def partition(rows, question):
  """Partitions a dataset.
  For each row in the dataset, check if it matches the question. If
  so, add it to 'true rows', otherwise, add it to 'false rows'.
  """
  true_rows, false_rows = [], []
  for row in rows:
    if question.match(row):
      true_rows.append(row)
    else:
      false_rows.append(row)
  return true_rows, false_rows

def cond_entropy(train_data):
  '''
  Caculate each conditional entropy.
  '''
  counts = label_counts(train_data)
  if 2 not in counts:
    return 0
  elif 4 not in counts:
    return 0
  else:
    p = float(counts[2]/(counts[2]+counts[4]))
    return -(p * math.log(p) + (1-p) * math.log(1-p))
  

def gini(train_data):
  '''
  Calculate the Gini Impurity for a list of rows.
  https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity
  '''
  counts = label_counts(train_data)
  impurity = 1
  for label in counts:
      prob_of_label = counts[label] / float(len(train_data))
      impurity -= prob_of_label**2
  return impurity

def find_best_split_entropy(rows):
  '''
  Find the best question to ask by iterating over every feature / value and calculating the information gain.
  '''
  best_entropy = 1  
  best_question = None
  n_features = len(rows[0]) - 1  # number of columns

  for col in range(n_features):  # for each feature
    question = Question(col, 7) # if the feature >= 7?
    # try splitting the dataset
    true_rows, false_rows = partition(rows, question)
    # Skip this split if it doesn't divide the dataset.
    if len(true_rows) == 0 or len(false_rows) == 0:
      continue
    p = float(len(true_rows)) / (len(true_rows) + len(false_rows))
    entropy = p * cond_entropy(true_rows) + (1 - p) * cond_entropy(false_rows)
    if entropy <= best_entropy:
        best_entropy, best_question = entropy, question
  return best_entropy, best_question

def find_best_split_gini(rows):
  '''
  Find the best question to ask by iterating over every feature / value and calculating the information gain.
  '''
  best_gain = 0  # keep track of the best gini impurity
  best_question = None
  current_uncertainty = gini(rows)
  n_features = len(rows[0]) - 1  # number of columns

  for col in range(n_features):  # for each feature
    question = Question(col, 7) # if the feature >= 7?
    # try splitting the dataset
    true_rows, false_rows = partition(rows, question)
    # Skip this split if it doesn't divide the dataset.
    if len(true_rows) == 0 or len(false_rows) == 0:
      continue
    p = float(len(true_rows)) / (len(true_rows) + len(false_rows))
    gain = current_uncertainty - p * gini(true_rows) - (1 - p) * gini(false_rows)
    if gain >= best_gain:
        best_gain, best_question = gain, question
  return best_gain, best_question

class Leaf:
    """A Leaf node classifies data.
    """
    def __init__(self, rows):
        self.predictions = label_counts(rows)
   

class Decision_Node:
  """A Decision Node asks a question.
  This holds a reference to the question, and to the two child nodes.
  """
  def __init__(self, question, true_branch, false_branch):
    self.question = question
    self.true_branch = true_branch
    self.false_branch = false_branch

def build_tree_entropy(rows, n_nodes):
  '''
  build a tree which has "n_nodes" nodes, that is, if n_nodes == 1, the tree has only one node.
  '''  
  gain, question = find_best_split_entropy(rows)
  if n_nodes == 0: 
    return Leaf(rows)

  true_rows, false_rows = partition(rows, question)
  n_nodes -= 1
  true_branch = build_tree_entropy(true_rows, n_nodes)
  ###print( "n_nodes: ", n_nodes)
  false_branch = build_tree_entropy(false_rows, n_nodes)
  return Decision_Node(question, true_branch, false_branch)

def build_tree_gini(rows, n_nodes):
  '''
  build a tree which has "n_nodes" nodes, that is, if n_nodes == 1, the tree has only one node.
  '''  
  gain, question = find_best_split_gini(rows)
  if n_nodes == 0: 
    return Leaf(rows)

  true_rows, false_rows = partition(rows, question)
  n_nodes -= 1
  true_branch = build_tree_gini(true_rows, n_nodes)
  ###print( "n_nodes: ", n_nodes)
  false_branch = build_tree_gini(false_rows, n_nodes)
  return Decision_Node(question, true_branch, false_branch)

def classify(row, node):
  '''
  Using majority vote for the rows that come to the leaf node.
  '''
  # Base case: we've reached a leaf
  if isinstance(node, Leaf):
    leaf_counts = node.predictions
    if 2 not in leaf_counts:
      return 4
    elif 4 not in leaf_counts:
      return 2
    elif leaf_counts[2] > leaf_counts[4]:
      return 2 
    else:
      return 4 

  if node.question.match(row):
    return classify(row, node.true_branch)
  else:
    return classify(row, node.false_branch)

def get_accuracy(data, tree):
  '''
  Caculate the accuracy of train/test data for one spliting node, two nodes, ...
  Store the #nodes-accuracy pair in a dict accuracy[#nodes]
  '''
  accuracy = 0.00 
  for row in data:
    actual_label = row[-1]
    predict_label = classify(row, tree)
    if actual_label == predict_label:
      accuracy += float(1/len(data))
  return accuracy  
  

datafile = 'hw2_question1.csv'
data = load_csv(datafile)
X = data[:, 0:9]
y = data[:, -1]

train_data = data[0:456, :]
X_train = train_data[:, 0:9]
y_train = train_data[:, -1]

test_data = data[457:682, :]
X_test = test_data[:, 0:9]
y_test = test_data[:, -1]

##Get accuracy of increasing node tree, with train and test data

#entropy_tree = {}
#accuracy_train_data = {}
#accuracy_test_data = {}
#for i in range(4):
#  entropy_tree[i+1] = build_tree_entropy(train_data, i+1)
#  accuracy_train_data[i+1] = get_accuracy(train_data, entropy_tree[i+1])
#  accuracy_test_data[i+1] = get_accuracy(test_data, entropy_tree[i+1])
#
#print(accuracy_train_data)
#print(accuracy_test_data)
#x, y = zip(*accuracy_train_data.items()) # unpack a list of pairs into two tuples
#plt.plot(x, y)
#x_test, y_test = zip(*accuracy_test_data.items()) # unpack a list of pairs into two tuples
#plt.plot(x_test, y_test, 'ro')
#plt.axis([0.5, 4.5, 0.7, 1])
#plt.xlabel('# of decision tree nodes')
#plt.ylabel('Train/Test accuracy')
#plt.title('Decision tree, split nodes by information entropy')
#plt.show()

gini_tree = {}
accuracy_train_data2 = {}
accuracy_test_data2 = {}
for i in range(4):
  gini_tree[i+1] = build_tree_gini(train_data, i+1)
  accuracy_train_data2[i+1] = get_accuracy(train_data, gini_tree[i+1])
  accuracy_test_data2[i+1] = get_accuracy(test_data, gini_tree[i+1])

print(accuracy_train_data2)
print(accuracy_test_data2)
x2, y2 = zip(*accuracy_train_data2.items()) # unpack a list of pairs into two tuples
plt.plot(x2, y2)
x_test2, y_test2 = zip(*accuracy_test_data2.items()) # unpack a list of pairs into two tuples
plt.plot(x_test2, y_test2, 'ro')
plt.axis([0.5, 4.5, 0.7, 1])
plt.xlabel('# of decision tree nodes')
plt.ylabel('Train/Test accuracy')
plt.title('Decision tree, split nodes by Gini impurity')
plt.show()


