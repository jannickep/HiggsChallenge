'''
Grid Search 2
Jannicke Pearkes
jpearkes@uvic.ca
Purpose: to perform a grid search on the hyper parameters for the logistic regression classifier
'''


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
bestscore = 100
best_learning_rate = -999
best_n_epochs = -999
best_batch_size = -999
iteration = 0
count = 0
print "START______________________________________________________________________"

class Parameter:
    def __init__(self, name, low_param, high_param, num_steps):
        self.name = name
        self.low_param = low_param
        self.high_param = high_param
        self.num_steps = num_steps
        if self.num_steps == 0:
            self.values = [self.low_param]
        if self.num_steps == 1: #exponential type
            self.values= list(10**exp for exp in range(self.low_param,self.high_param))
        elif self.num_steps > 1: #linear type goes from low to just below high
             step_size = (self.high_param-self.low_param)*1.0/ self.num_steps
             self.values = list(self.low_param+step_size*i for i in range(0,self.num_steps)) 
            
    def __str__(self):
        string = self.name +','+ str(self.values)
        return str(string)
        
x = dict()
# Parameter( name, low_param, high_param, num_steps)
# num steps = 0 for constant value
#             1 for exponential increases
#             >1 for number of steps 
x[0] = Parameter("learning_rate", 0, 3, 1)
x[1] = Parameter("n_epochs", 0, 3, 1)
x[2] = Parameter("batch_size", 0, 3, 1)
x_best = [0,0,0]
# TODO: Put into for loop, but must be somewhat careful
# TODO: have sgd run without output
# TODO: Only load data once
print x[0]
print x[1]
print x[2]
length = len (x[0].values)*len (x[1].values)*len (x[2].values)
print length
mat = np.zeros((length,4))

for i in range(len (x[0].values)):
    for j in range(len (x[1].values)):
        for k in range(len (x[2].values)):
            score = sgd_optimization_mnist(x[0].values[i],
                         x[1].values[j],
                         x[2].values[k])
            if (score < bestscore and score != 0.0):
                x_best[0] = x[0].values[i]
                x_best[1] = x[1].values[j]
                x_best[2] = x[2].values[k]
            mat[count,:] = [x[0].values[i],x[1].values[j],x[2].values[k], score]
            count = count+1
            
            
print "best score of:"+str(bestscore)+"was obtained with the following parameters:"
print x[0].name +": "+ str(x_best[0])
print x[1].name +": "+ str(x_best[1])
print x[2].name +": "+ str(x_best[2])

cPickle.dump( mat, open( "grid_search_results.p", "wb" ) )


                
                         


