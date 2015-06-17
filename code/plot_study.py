# Author: Jannicke Pearkes
# Purpose: Makes plots of hyperparameters all on one page (looks messy)

import cPickle
import gzip
import os
import sys
import time
import random,string,math,csv
import scipy.linalg.blas
import numpy
import numpy as np
import numpy.lib.recfunctions
import matplotlib.pyplot as plt
import matplotlib as mpl
import operator

mpl.rcParams.update({'font.size': 6})

parameter_names = ["improvement_threshold", "learning_rate", "batch_size", "n_epochs", "patience", "submit_threshold", "patience_increase", "L1_reg", "n_hidden", "L2_reg", "test_score", "test_std_dev", "walltime"]
#parameter_names = np.fromstring(parameter_string,dtype = str, sep=',')


array = cPickle.load(open("array.p",'rb'))

# p = np.zeros((len(array),1))
# y = np.zeros((len(array),1))
# t = np.zeros((len(array),1))

# for i in range(len(array)):
#     p[i] = reduce(operator.mul,array[i][:-2], 1) 
# y = array[:,-2]
# t = array[:,-1]

# miny = np.argwhere(y == np.amin(y))
# print ("min-y:" + str( miny))
# for i in range(len(miny)):
#     print ("x at min" + str(array[miny[i],:]))

# #print "size x"
# #print p.size
# #print p
# #print "size y"
# #print y.size
# #print y
name = parameter_names[0]
values = array[:,0]
validation_error = array[:,-3]
validation_uncertainty = array[:,-2]
time = array[:,-1]

plt.figure(0)
plt.subplot(2,1,1)
plt.errorbar(values,validation_error, validation_uncertainty, linestyle = "None")

plt.title("  ")
plt.xlabel(parameter_names[0])
plt.ylabel(parameter_names[-3])

plt.subplot(2,1,2)
plt.scatter(values,time)

plt.title("   ")
plt.xlabel(parameter_names[0])
plt.ylabel(parameter_names[-1])
plt.savefig("../plots/improvement_threshold_study")
'''
plt.figure(1)

for i in range(7):
    plt.subplot(3,3,i+1)
    plt.scatter(array[:,i],y)
    plt.xlabel(x[i].name)
    plt.ylabel("Test Error %")
    plt.savefig("../plots/errorplots")

plt.figure(2)
for i in range(7):
    plt.subplot(3,3,i+1)
    plt.scatter(array[:,i],t)
    plt.xlabel(x[i].name)
    plt.ylabel("Time(min)")
    plt.savefig("../plots/timeplots")
'''