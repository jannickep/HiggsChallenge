import string
import subprocess
import sys
from collections import OrderedDict
import numpy as np
# Mlp Optimization 
# Purpose: To submit jobs for the mlp to the batch scheduler
#          Ideally will work with any of the NN algorithms
'''
class Parameter:
  def __init(self, name, values):
    self.name = name
    self.values = values


if(len(sys.argv) > 1):
      input = sys.argv[1]
      if(input == "mlp"):
            parameters = dict(
                  learning_rate = 0.998,
                  L1_reg = 0.00005,
                  L2_reg = 0.000005,
                  n_hidden = 50,
                  n_epochs = 10000,
                  batch_size = 600,
                  patience = 10000,
                  #patience = 10,
                  patience_increase = 2,
                  #improvement_threshold = 0.05,
                  improvement_threshold = 0.995,
                  submit_threshold = 0.5
                  )
            pbs_name = " mlp_batch.pbs"
      elif(input == "sda"):
            parameters = dict(
                 improvement_threshold = [0.995],
                 finetune_lr = [1,0.1,0.01],
                 pretraining_epochs = [10],
                 pretrain_lr = [0.001], 
                 training_epochs = [500],
                 batch_size = [1],
                 neurons_per_layer = [20,30,40],
                 number_of_layers = [1,2,3,4],
                 patience = [10000],
                 patience_increase = [2],
                 submit_threshold = [0.5]
                  )
            pbs_name = " SdA_batch.pbs"
      elif(input == "log"):
            parameters = dict(
                  learning_rate = 0.998, 
                  n_epochs = 10000,
                  batch_size = 600,
                  patience = 5000, 
                  patience_increase = 2,
                  improvement_threshold = 0.995,
                  submit_threshold = 0.5)
            pbs_name = " log_batch.pbs"
else:
      print "No input specified, defaulting to single layer NN"
      parameters = dict(
            learning_rate = 0.998,
            L1_reg = 0.00005,
            L2_reg = 0.000005,
            n_hidden = 50,
            n_epochs = 10000,
            batch_size = 600,
            patience = 10000,
            patience_increase = 2,
            improvement_threshold = 0.995,
            submit_threshold = 0.5
            )
      pbs_name = " mlp_batch.pbs"
parameters = OrderedDict(sorted(parameters.items()))
print "START OPTIMIZATION WITH ______________________________________________________________________"

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

import subprocess

#TODO: add in matrix rows
#TODO: add in best value checker
#TODO: would need output to go somewhere proper 

bestscore = 100
best_learning_rate = -999
best_n_epochs = -999
best_batch_size = -999
iteration = 0
count = 0
print "START______________________________________________________________________"

class Parameter:
    def __init__(self, name, values):
        self.name = name
        self.values = values
                   
    def __str__(self):
        string = self.name +','+ str(self.values)
        return str(string)
        
x = dict()
# Parameter( name, low_param, high_param, num_steps)
# num steps = 0 for constant value
#             1 for exponential increases
#             >1 for number of steps
#finetune_lr,pretraining_epochs,pretrain_lr,training_epochs,batch_size,neurons_per_layer,number_of_layers 

x[0] = Parameter("L1_reg" , [0.00005])
x[1] = Parameter("L2_reg" , [0.000005])
x[2] = Parameter("batch_size" , [1, 10, 100, 1000])
x[3] = Parameter("improvement_threshold" , [0.995])
x[4] = Parameter("n_epochs" , [500, 1000])
x[5] = Parameter("n_hidden" , [20,30,40,50])
x[6] = Parameter("learning_rate" , [1,0.1,0.01])
x[7] = Parameter("patience" , [1000,10000])
x[8] = Parameter("patience_increase" , [2])
x[9]= Parameter("submit_threshold" , [0.5])
'''
#tester
x[0] = Parameter("L1_reg" , [0.00005])
x[1] = Parameter("L2_reg" , [0.000005])
x[2] = Parameter("batch_size" , [1])
x[3] = Parameter("improvement_threshold" , [0.995])
x[4] = Parameter("learning_rate" , [1])
x[5] = Parameter("n_epochs" , [500])
x[6] = Parameter("n_hidden" , [20])
x[7] = Parameter("patience" , [1000])
x[8] = Parameter("patience_increase" , [2])
x[9]= Parameter("submit_threshold" , [0.5])
'''
x_best = [0,0,0]

# TODO: this is a hack, worked ok with only 3 hyper-parameters. 
#       Use recursion instead

length = 1
for i in range(0,10):
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
                          for p in range(len (x[7].values)):
                            for q in range(len (x[8].values)):
                              for r in range(len (x[9].values)):
                                  talk = ('qsub -vinput1='+str(x[0].values[i])
                                              +',input2='+str(x[1].values[j])
                                              +',input3='+str(x[2].values[k])
                                              +',input4='+str(x[3].values[l])
                                              +',input5='+str(x[4].values[m])
                                              +',input6='+str(x[5].values[n])
                                              +',input7='+str(x[6].values[o])
                                              +',input8='+str(x[7].values[p])
                                              +',input9='+str(x[8].values[q])
                                              +',input10='+str(x[9].values[r])
                                              +' mlp_batch.pbs')
                                  print talk
                                  subprocess.call(talk, shell = True)
print "talking complete" 
