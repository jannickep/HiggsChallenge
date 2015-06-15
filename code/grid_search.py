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
from logistic_sgd import LogisticRegression, load_data, sgd_optimization_mnist

'''
Hyper Parameters for logistic regression:
    learning_rate = .13
    n_epochs = 1000
    batch_size = 600
    patience = 5000  
    patience_increase = 2  
    improvement_threshold = 0.995 
    
    Plan to optimize on test loss
    Plan to plot variations
    Save maxima?
    Power search from 10^-2 10^4? 
'''
best_test_score = 100
best_learning_rate = -999
best_n_epochs = -999
best_batch_size = -999
iteration = 0
print "START______________________________________________________________________"
batch_size_number_list = (10**exp for exp in range(0,6))
n_epochs_number_list = (10**exp for exp in range(1,6))
learning_rate_number_list = (10**exp for exp in range(-2,6))
print "batch_size_number_list"+(str(batch_size_number_list))
print "n_epochs_number_list"+(str(n_epochs_number_list))
print "learning_rate_number_list"+(str(learning_rate_number_list))

for learning_rate in learning_rate_number_list:
    for n_epochs in n_epochs_number_list:
        for batch_size in batch_size_number_list:
            print "learning rate:"
            print learning_rate
            print "n epochs"
            print n_epochs
            print "batch size:"
            print batch_size
            iteration += iteration
            print "iteration" 
            print iteration 
            test_score = sgd_optimization_mnist(learning_rate=learning_rate,       
                                                n_epochs=n_epochs,
                                                batch_size=batch_size)
            if (test_score < best_test_score) and (test_score != 0):
                best_test_score = test_score
                best_learning_rate = learning_rate
                best_n_epochs = n_epochs
                best_batch_size = batch_size
print "overall best performance with test score of:"
print best_test_score
print "learning rate:"
print learning_rate
print " n epochs:"
print n_epochs
print "batch size:"
print batch_size

#TODO: For tomorrow turn the parameters into a list of parameters! =)
                
                
            

