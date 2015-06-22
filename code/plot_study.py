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
import labels_dictionary as ld
from scipy.optimize import curve_fit

# Labels dictionary contains nice versions of parameter names to use on graphs
def label(string):
    return ld.labels.get(string)
# Use fontsie 10 in plots
mpl.rcParams.update({'font.size': 10})
# Import parameter names for the model
header = cPickle.load(open("header.p",'rb'))
parameter_names = header.split(",")
print parameter_names
# Import parsed output data
array = cPickle.load(open("array.p",'rb'))

num_col = 3 # The column we want to study
name = parameter_names[num_col]
values = array[:,num_col]
validation_error = array[:,-3]
validation_uncertainty = array[:,-2]
time = array[:,-1]
time_uncertainty = [1/120.0 for t in time]

# Find min and max values and pad a little for plot axes
pad_u = 1.1
pad_d = 0.9
v_min = min(values)*pad_d
v_max = max(values)*pad_u
e_min = min(validation_error)*pad_d
e_max = max(validation_error)*pad_u
t_min = min(time)*pad_d
t_max = max(time)*pad_u
print v_min,v_max
print e_min,e_max
print t_min,t_max

# Linear fitting function 
def func(x, p1,p2):
  return p1*x + p2

plt.figure(0)
plt.subplot(2,1,1)
plt.title("Effect of Patience Hyper-Parameter on Logistic Regression")
plt.errorbar(values,validation_error, validation_uncertainty, linestyle = "None")
plt.xlabel(label(name))
plt.ylabel(label(parameter_names[-3]))
plt.ylim(e_min,e_max)
plt.xlim(v_min,v_max)
# Add fitted line to plot of percent error
popt,pcov = curve_fit(func, values, validation_error, sigma = validation_uncertainty)
[a,b] = popt
x_new = np.linspace(v_min,v_max,num =10)
y_new = func(x_new, a,b)
plt.plot(x_new,y_new)
print popt # this has slope and y-intercept of our line

# Time plot
plt.subplot(2,1,2)
plt.scatter(values,time,edgecolors = 'none')
#plt.errorbar(values,time, time_uncertainty, linestyle = "None") # Errorbar is so tiny that I am not plotting it
plt.title("   ") # Space at top
plt.xlabel(label(name))
plt.ylabel(label(parameter_names[-1]))
plt.xlim((v_min,v_max))
# Add linear fit
popt,pcov = curve_fit(func, values, time, sigma = time_uncertainty)
[a,b] = popt
x_new = np.linspace(v_min,v_max,num =10)
y_new = func(x_new, a,b)
print popt
plt.plot(x_new,y_new)
plt.savefig("../plots/"+name+"_study_sda")


