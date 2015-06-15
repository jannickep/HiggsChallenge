# Author: Jannicke Pearkes
# Purpose: Normalizes and shuffles training data for the Higgs Challenge 


import os
import sys
import time

import random
import string
import math
import csv
import scipy.linalg.blas
import numpy as np
import cPickle

def pre_process_data():
    #Percentage for Training, Validation and Test Sets
    train_p = .80
    valid_p = .90
    test_p = 1

    #Read in Higgs Data Set
    print "Reading training set ..."
    with open("../data/training.csv","rb") as f:
        raw_data = list(csv.reader( f, delimiter=',')) 
    header = list(raw_data[0])
    raw_data = raw_data[1:]

    print "Shuffling data ... "
    np.random.shuffle(raw_data) # Shuffle data set

    # Assign inputs (x_raw), outputs(y_raw) and weights (w_raw)
    x_raw = np.array([map(float, row[1:-2]) for row in raw_data[:]])
    y_raw = np.array([map(str, row[-1]) for row in raw_data[:]])
    w_raw = np.array([float(row[-2]) for row in raw_data[:]])

    # Set 's' inputs to 1 and 'b' inputs to 0
    y_raw = np.where(y_raw == 's', 1,0).flatten()
    w_raw = w_raw.flatten()

    # Normalize inputs
    print "Normalizing data ..."
    m,n = x_raw.shape
    print 'x_raw shape:'
    print  x_raw.shape

    #Initialize Arrays to 0
    sum_x = np.zeros(n) 
    count = np.zeros(n)
    mean = np.zeros(n)
    std_dev = np.zeros(n)
    x_new = np.zeros((m,n))
    x_calc = np.zeros((m,n))

    # Input valid data (not -999) into array
    for col in range(0,n):
        for row in range(0,m):
            if (x_raw[row][col] != -9.99000000e+02):
                 x_calc[count[col]][col] = x_raw[row][col]
                 count[col] = count[col]+1 #keep track of number of valid data points

    #print 'x_calc:' 
    #print repr(x_calc)
    #print repr(count)
    #print 'x_raw:' 
    #print repr(x_raw)

    # Calculate mean and standard deviation for valid data
    for n in range (0,n):
        mean[n] = np.mean(x_calc[0:count[n],n])
        std_dev[n] = np.std(x_calc[0:count[n],n])
    print 'mean'
    print repr(mean)
    print 'std_dev'
    print repr(std_dev)

    # Use mean and standard deviation to normalize data
    for col in range(0,n):
        for row in range(0,m):
            if ((x_raw[row][col] != -9.99000000e+02) and (std_dev[col]!= 0)): # std_dev being zero would create a nan
                x_new[row][col] = (x_raw[row][col] - mean[col]) / std_dev[col] 
    print repr(x_new)

    # Create tuples for use with theano
    print "Creating tuples"
    train_set_x = x_new[0:int(m*train_p)]
    valid_set_x = x_new[int(m*train_p):int(m*valid_p)]
    test_set_x = x_new[int(m*valid_p):int(m*test_p)]

    train_set_y = y_raw[0:int(m*train_p)]
    valid_set_y = y_raw[int(m*train_p):int(m*valid_p)]
    test_set_y = y_raw[int(m*valid_p):int(m*test_p)]

    train_set_w = w_raw[0:int(m*train_p)]
    valid_set_w = w_raw[int(m*train_p):int(m*valid_p)]
    test_set_w = w_raw[int(m*valid_p):int(m*test_p)]

    train_set = train_set_x, train_set_y, train_set_w 
    valid_set = valid_set_x, valid_set_y, valid_set_w
    test_set = test_set_x, test_set_y, test_set_w

    data = train_set, valid_set, test_set

    print "Saving data"
    with  open("../data/normalizedTraining.p", "wb") as f:
        cPickle.dump( data, f )

    print "Preprocessing of training set complete"

if __name__ == '__main__':
    pre_process_data()
