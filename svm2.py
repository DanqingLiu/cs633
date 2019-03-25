'''
https://www.csie.ntu.edu.tw/~cjlin/libsvm/
https://stackoverflow.com/questions/89228/calling-an-external-command-in-python
'''
import csv
import random
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from subprocess import *

import time

def load_csv(filename):
  '''
  load data.
  '''
  lines = csv.reader(open(filename, 'rt', encoding = 'utf-8'))
  dataset = list(lines)
  result = np.array(dataset).astype("int")
  np.random.shuffle(result) # randomly re-arrange the rows of the data samples
  return result

def preprocessing(data):
    """one-hot encoded features."""
    return np.eye(3)[data]

data_file = load_csv('hw2_question3.csv')
print(data_file.shape)

train_data = data_file[0:7370, :]
X_train = train_data[:,0:30]
y_train = train_data[:,-1]
test_data = data_file[7370:11055, :]
X_test = test_data[:,0:30]
y_test = test_data[:,-1]

###Data preprocessing for one hot code
X_train_pre = preprocessing(X_train)
X_test_pre = preprocessing(X_test)

###Data preprocessing for libsvm
def for_libsvm(data, f):
  for row in data:
    new_line = []
    new_line.append(str(row[-1]))
    for i in range(30):
      new_item = "%s:%s" % ( i + 1, row[i])
      new_line.append(new_item)
    new_line = " ".join(new_line)
    new_line += "\n"
    f.write(new_line)
  
f_train = "train_file_rbf.scale"
f = open(f_train, "a")
for_libsvm(train_data, f)
f.close()

#start = time.time()
#scaled_file = "train_file_poly.scale"
#model_file = "train_file_poly.model"
#t = 1
#c = 10
#svmtrain_exe = "./libsvm/svm-train"
#cmd = '{0} -c {1} -t {2} "{3}" "{4}"'.format(svmtrain_exe,c,t,scaled_file,model_file)
#print('Training...')
#Popen(cmd, shell = True, stdout = PIPE).communicate()
#
#end = time.time()
#print("Training time for C = 10: ", end - start)
#
#os.system("./libsvm/svm-predict test_file train_file_poly.model output_file")

start = time.time()
scaled_file = "train_file_rbf.scale"
model_file = "train_file_rbf.model"
t = 2
c = 10
svmtrain_exe = "./libsvm/svm-train"
cmd = '{0} -c {1} -t {2} "{3}" "{4}"'.format(svmtrain_exe,c,t,scaled_file,model_file)
print('Training...')
Popen(cmd, shell = True, stdout = PIPE).communicate()

end = time.time()
print("Training time for C = 10: ", end - start)

os.system("./libsvm/svm-predict test_file train_file_rbf.model output_file")
