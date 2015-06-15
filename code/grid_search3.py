'''
Grid Search 3
Author: Jannicke Pearkes
jpearkes@uvic.ca
Purpose: submit jobs to a batch scheduler to perform a grid search on the hyper parameters for the stacked denoising autoencoder
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
import subprocess

#TODO: add in matrix rows
#TODO: add in best value checker
#TODO: would need output to go somewhere proper 
               
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
             step_size = (self.high_param-self.low_param)/ self.num_steps
             self.values = list(self.low_param+step_size*i for i in range(0,self.num_steps)) 
            
    def __str__(self):
        string = self.name +','+ str(self.values)
        return str(string)
        
x = dict()
# Parameter( name, low_param, high_param, num_steps)
# num steps = 0 for constant value
#             1 for exponential increases
#             >1 for number of steps
#finetune_lr,pretraining_epochs,pretrain_lr,training_epochs,batch_size,neurons_per_layer,number_of_layers 
x[0] = Parameter("finetune_lr", 0.015, .02, 0)
x[1] = Parameter("pretraining_epochs", 10, 20, 0)
x[2] = Parameter("pretrain_lr", 0.0015, .002, 0)
x[3] = Parameter("training_epochs", 500, 2000, 0)
x[4] = Parameter("batch_size", 1, 2, 0)
x[5] = Parameter("neurons_per_layer",20,60,4)
x[6] = Parameter("number_of_layers",2,4,0)


'''
x[0] = Parameter("finetune_lr", 0.5, .2, 3)
x[1] = Parameter("pretraining_epochs", 10, 25, 3)
x[2] = Parameter("pretrain_lr", 0.0005, .002, 3)
x[3] = Parameter("training_epochs", 500, 2000, 3)
x[4] = Parameter("batch_size", 1, 2, 1)
x[5] = Parameter("neurons_per_layer",500,2000,3)
x[6] = Parameter("number_of_layers",1,4,3)


x[0] = Parameter("finetune_lr", 0, 3, 1)
x[1] = Parameter("pretraining_epochs", 0, 3, 1)
x[2] = Parameter("pretrain_lr", 0, 3, 1)
x[3] = Parameter("training_epochs", 0, 3, 1)
x[4] = Parameter("batch_size", 0, 1, 1)
x[5] = Parameter("neurons_per_layer",0,2,1)
x[6] = Parameter("number_of_layers",3,5,0)


original
x[0] = Parameter("finetune_lr", 0.15, 3, 0)
x[1] = Parameter("pretraining_epochs", 15, 3, 0)
x[2] = Parameter("pretrain_lr", 0.001, 3, 0)
x[3] = Parameter("training_epochs", 1000, 3, 0)
x[4] = Parameter("batch_size", 1, 2, 0)
x[5] = Parameter("neurons_per_layer",1000,2,0)
x[6] = Parameter("number_of_layers",3,5,0)
'''
x_best = [0,0,0]
# TODO: Put into for loop, but must be somewhat careful
# TODO: have sgd run without output
# TODO: Only load data once
length =1
for i in range(0,6):
    print x[i]
    length = length*len(x[i].values)
print "number of iterations: "+str(length)

print "beginning to talk" 
for i in range(len (x[0].values)):
    for j in range(len (x[1].values)):
        for k in range(len (x[2].values)):
            for l in range(len (x[3].values)):
                for m in range(len (x[4].values)):
                    for n in range(len (x[5].values)):
                        for o in range(len (x[6].values)):
                            talk = ('qsub -vinput1='+str(x[0].values[i])
                                        +',input2='+str(x[1].values[j])
                                        +',input3='+str(x[2].values[k])
                                        +',input4='+str(x[3].values[l])
                                        +',input5='+str(x[4].values[m])
                                        +',input6='+str(x[5].values[n])
                                        +',input7='+str(x[6].values[o])
                                        +' SdA_batch.pbs')
                            print talk
                            subprocess.call(talk, shell = True)
print "talking complete" 
