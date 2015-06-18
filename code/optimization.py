import string
import subprocess
import sys
from collections import OrderedDict
import numpy as np
# Mlp Optimization 
# Purpose: To submit jobs for the mlp to the batch scheduler
#          Ideally will work with any of the NN algorithms
LINEAR = 0
EXPONENTIAL = 0
CHANGES = 1
PERCENT = 5
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
                 improvement_threshold = 0.995,
                 finetune_lr = 0.1,
                 pretraining_epochs = 15,
                 pretrain_lr = 0.001, 
                 training_epochs = 1000,
                 batch_size = 1,
                 neurons_per_layer = 32,
                 number_of_layers = 3,
                 patience = 10000,
                 patience_increase = 2,
                 submit_threshold = 0.5
                  )
      elif(input == "log"):
            parameters = dict(
                  learning_rate = 0.13, 
                  n_epochs = 1000,
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

# for key in Parameters
# if value needs to be changed, change
# qsub only item that need to be changed 

# how do i decide on whether an item needs to be changed or not?
# give it an array
# create array for each parameter, just use one values if does not change

# Could have user input? Maybe if start with mlp_optimization.py i <- interactive 
# Number of parameters will be different if using mlp over log_sgd

# how do I input what I would like

# Parameter to be changed
# How to change it 
'''
print "which parameter would you like to adjust?"
i = 0
for key,value in parameters.items():
    print (str(i)+") "+ key+":"+str(value))
    i += 1
'''
# Easiest just to put in percent change that would like to see 
# instead of actual values
# automatically do linear 

# Just do 20% change for all parameters default?
# A generic talk function for qsub
#iter_key = "improvement_threshold"
#iter_key = "batch_size"
iter_key = "patience"
difference = float(parameters[iter_key]*PERCENT/100.0)
newvalue = parameters[iter_key] - int(CHANGES/2)*difference 
centre = parameters[iter_key]
num = 2*CHANGES+1
parameter_updates = np.zeros(num)
if (EXPONENTIAL):
    start = centre*(10**(-CHANGES))
    stop = centre*(10**CHANGES)
    print "Exponential: logspace(start = "+str(start)+", stop = "+str(stop)+", num = "+str(num)+")"
    for i in range(num):
      parameter_updates[i] = start*(10**i)
else:
    start = centre-CHANGES*PERCENT
    stop = centre+CHANGES*PERCENT
    parameter_updates = np.linspace(start,stop, num = num)
    parameter_updates = np.linspace(1,100000, num = 10)
for j in parameter_updates:
      parameters[iter_key] = j
      i = 1
      talk = "qsub -v"
      for key,value in parameters.items():
            print key+": "+str(value)
            talk += "input"+str(i)+"="+str(value)+","
            i += 1
      talk = talk[:-1]
      talk += pbs_name
      print talk 
      subprocess.call(talk, shell = True)
      newvalue = parameters[iter_key] + difference

#print "Submitting "+number_of_jobs+" jobs ..."
