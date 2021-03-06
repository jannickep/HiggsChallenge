#Author: Jannicke Pearkes
#Purpose: Normalizes and weights the test data set for the Higgs Challenge

import os
import sys
import time

import numpy

import theano
import theano.tensor as T
import random,string,math,csv
import scipy.linalg.blas
import numpy as np
import theano.tensor as T
import cPickle



raw_data = list(csv.reader(open("../data/test.csv","rb"), delimiter=','))
num_rows = len(raw_data)-1
header = list(raw_data[0])

x_raw = np.array([map(float, row[1:]) for row in raw_data[1:]])
test_id = np.array([int(row[0]) for row in raw_data[1:]])


print "Normalizing ..."
m,n = x_raw.shape
print 'x_raw shape:'
print  x_raw.shape
sum_x = np.zeros(n)
count = np.zeros(n)
mean = np.zeros(n)
std = np.zeros(n)
x_new = np.zeros((m,n))
x_calc = np.zeros((m,n))
#print count

# input valid data into array
for col in range(0,n):
    for row in range(0,m):
        if (x_raw[row][col] != -9.99000000e+02):
             x_calc[count[col]][col] = x_raw[row][col]
             count[col] = count[col]+1
#print 'x_calc:' 
#print repr(x_calc)
#print repr(count)
#print 'x_raw:' 
#print repr(x_raw)


# calculate mean and standard deviation for valid data
for n in range (0,n):
    mean[n] = np.mean(x_calc[0:count[n],n])
    std[n] = np.std(x_calc[0:count[n],n])
print 'mean'
print repr(mean)
print 'std'
print repr(std)


for col in range(0,n):
    for row in range(0,m):
        if ((x_raw[row][col] != -9.99000000e+02) and (std[col]!= 0)): # std being zero would create a nan
            x_new[row][col] = (x_raw[row][col]-mean[col])/std[col] 
print repr(x_new)
           

test_set_x = x_new



test_set = test_set_x, test_id

data = test_set
cPickle.dump( data, open( "test_data.p", "wb" ) )

'''
submission = np.array(([str (','.join (map (str, (x_new[tI,:]) ) ) ),
                            str(w_raw[tI]),
                            str(','.join(map(str, y_raw[tI])))]  
                            for tI in range(len(y_raw))))
print("header is being added")
print 'header dim' 
headery = np.asarray(header)
print headery.shape
print 'submission dim' 
print submission.shape 
submission = np.append([np.asarray(header)], submission, axis=0)
print('submission is being saved')
np.savetxt("test_processed.csv",submission,fmt='%s',delimiter=',')
'''
print('complete')
