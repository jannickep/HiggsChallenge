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
#finetune_lr,pretraining_epochs,pretrain_lr,training_epochs,batch_size,neurons_per_layer,number_of_layers 

x[0] = Parameter("finetune_lr", 0, 3, 1)
x[1] = Parameter("pretraining_epochs", 0, 3, 1)
x[2] = Parameter("pretrain_lr", 0, 3, 1)
x[3] = Parameter("training_epochs", 0, 3, 1)
x[4] = Parameter("batch_size", 0, 1, 1)
x[5] = Parameter("neurons_per_layer",0,2,1)
x[6] = Parameter("number_of_layers",3,5,0)


array = cPickle.load(open("array.p",'rb'))
p = np.zeros((len(array),1))
y = np.zeros((len(array),1))
t = np.zeros((len(array),1))

for i in range(len(array)):
    p[i] = reduce(operator.mul,array[i][:-2], 1) 
y = array[:,-2]
t = array[:,-1]

miny = np.argwhere(y == np.amin(y))
print ("min-y:" + str( miny))
for i in range(len(miny)):
    print ("x at min" + str(array[miny[i],:]))

#print "size x"
#print p.size
#print p
#print "size y"
#print y.size
#print y
plt.figure(0)
plt.subplot(2,1,1)
plt.scatter(p,y)

plt.title("  ")
plt.xlabel("Product of Hyper Parameters")
plt.ylabel("Test Error %")

plt.subplot(2,1,2)
plt.scatter(t,y)

plt.title("   ")
plt.xlabel("Time(s)")
plt.ylabel("Test Error (%)")
plt.savefig("../plots/first_plots2")

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
