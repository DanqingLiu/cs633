import numpy as np
from scipy import signal

def myconv2d(image, filterr):
    m = filterr.shape[0]
    y, x = image.shape
    y = y - m + 1
    x = x - m + 1
    conv = np.zeros((y,x))    
    for i in range(y):
        for j in range(x):
            conv[i][j] = np.sum(image[i:i+m, j:j+m]*filterr)
    return conv
 
image = np.array([[164,188,164,161,195],
                  [178,201,197,150,137],
                  [174,168,181,190,184],
                  [131,179,176,185,198],
                  [92,185,179,133,167]
                 ])
filter1 = np.array([[1,1,1],[1,1,1],[1,1,1]])
filter2 = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
filter3 = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
#conv1 = signal.convolve2d(image, filter1, mode='valid')
#print(conv1)

conv_f1 = myconv2d(image, filter1)
conv_f2 = myconv2d(image, filter2)
conv_f3 = myconv2d(image, filter3)
print(conv_f1)
print(conv_f2)
print(conv_f3)
