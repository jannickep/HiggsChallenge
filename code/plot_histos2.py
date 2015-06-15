t__ = 'restructedtext en'

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
#import output_parser
#from plot_hyper_parameters import Parameter

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
x[0] = Parameter("Finetuning Learning Rate", 0.0005, .02, 3)
x[1] = Parameter("Number of Pretraining Epochs", 5, 20, 3)
x[2] = Parameter("Pretraining Learning Rate", 0.0005, .002, 3)
x[3] = Parameter("Number of Training Epochs", 500, 2000, 3)
x[4] = Parameter("Batch Size", 1, 2, 0)
x[5] = Parameter("Neurons Per Layer",50,1550,3)
x[6] = Parameter("Number of Layers",1,4,3)

q = int(sys.argv[1])
#q = [1,2,3,4,6]
t = int(sys.argv[2])

array = cPickle.load(open("array.p",'rb'))
m,n = array.shape
layer = []
line = []
l = len(x[q].values)
print m
for i in range(len(x[q].values)): 
   for j in range(m):
        if (array[j,q] == x[q].values[i]):
            if(t == 0):
                line.append(array[j,-1])
            if(t == 1):
                line.append(array[j,-2])
   layer.append(line)
   line = []        
   #print layer
c_max = max([max(layer[0]),max(layer[1]),max(layer[2])])
c_min = min([min(layer[0]),min(layer[1]),min(layer[2])])
#c_max = max([layer[1].max(),layer[2].max(),layer[3].max()])
#c_min = min([layer[1].min(),layer[2].min(),layer[3].min()])
print "c_max"+str(c_max)
print "c_min"+str(c_min)

# Get histograms 
Histo_layer1 = np.histogram(layer[0],bins=20,range=(c_min,c_max))
Histo_layer2 = np.histogram(layer[1],bins=20,range=(c_min,c_max))
Histo_layer3 = np.histogram(layer[2],bins=20,range=(c_min,c_max))
  
# Lets get the min/max of the Histograms
AllHistos= [Histo_layer1,Histo_layer2,Histo_layer3]
h_max = max([histo[0].max() for histo in AllHistos])*1.2
# h_min = max([histo[0].min() for histo in AllHistos])
h_min = 0.0
  
# Get the histogram properties (binning, widths, centers)
bin_edges = Histo_layer1[1]
bin_centers = ( bin_edges[:-1] + bin_edges[1:]  ) /2.
bin_widths = (bin_edges[1:] - bin_edges[:-1])
  
# To make error bar plots for the data, take the Poisson uncertainty sqrt(N)
#ErrorBar_testing_A = np.sqrt(Histo_testing_A[0])
# ErrorBar_testing_B = np.sqrt(Histo_testing_B[0])
  
# Draw objects
ax1 = plt.subplot(111)
  
# Draw solid histograms for the training data
ax1.bar(bin_centers-bin_widths/2.,Histo_layer1[0],facecolor='red',linewidth=0,width=bin_widths,label=(x[q].name+'='+str(x[q].values[0])),alpha=0.5)
ax1.bar(bin_centers-bin_widths/2.,Histo_layer2[0],facecolor='blue',linewidth=0,width=bin_widths,label=(x[q].name+'='+str(x[q].values[1])),alpha=0.5)
ax1.bar(bin_centers-bin_widths/2.,Histo_layer3[0],facecolor='black',linewidth=0,width=bin_widths,label=(x[q].name+'='+str(x[q].values[2])),alpha=0.5)
#ff = (1.0*(sum(Histo_layer1[0])+sum(Histo_layer2[0])))/(1.0*sum(Histo_layer2[0]))
 
# # Draw error-bar histograms for the testing data
#ax1.errorbar(bin_centers, ff*Histo_testing_A[0], yerr=ff*ErrorBar_testing_A, xerr=None, ecolor='black',c='black',fmt='.',label='Test (reweighted)')
# ax1.errorbar(bin_centers, Histo_testing_B[0], yerr=ErrorBar_testing_B, xerr=None, ecolor='red',c='red',fmt='o',label='B (Test)')
  
# Make a colorful backdrop to show the clasification regions in red and blue
#ax1.axvspan(pcut, c_max, color='blue',alpha=0.08)
#ax1.axvspan(c_min,pcut, color='red',alpha=0.08)
  
# Adjust the axis boundaries (just cosmetic)
#ax1.axis([c_min, c_max, h_min, h_max])
  
# Make labels and title
if t == 0:
    plt.title("Time to Run at Different Values of "+x[q].name)
    plt.xlabel("Time (min)")
if t == 1:
    plt.title("Test Errors at Different Values of "+x[q].name)
    plt.xlabel("Percent (%)")

plt.ylabel("Counts/Bin")
 
# Make legend with smalll font
legend = ax1.legend(loc='upper right', shadow=False,ncol=1)
for alabel in legend.get_texts():
            alabel.set_fontsize('small')
  
# Save the result to png
if t== 0:
    plt.savefig("../plots/histos_time_top5_"+x[q].name+".png")
if t == 1:
    plt.savefig("../plots/histos_error_top5_"+x[q].name+".png")


'''
histogram = np.histogram(x_raw[:,], 20)
mpl.rcParams.update({'font.size': 4})

num_bins = 100
x = dict()
x[0] = Parameter("finetune_lr", 0.0005, .02, 3)
x[1] = Parameter("pretraining_epochs", 5, 20, 3)
x[2] = Parameter("pretrain_lr", 0.0005, .002, 3)
x[3] = Parameter("training_epochs", 500, 2000, 3)
x[4] = Parameter("batch_size", 1, 2, 0)
x[5] = Parameter("neurons_per_layer",50,1550,3)
x[6] = Parameter("number_of_layers",1,4,3)

array = cPickle.load(open("array.p",'rb'))

# set missing values to 0 to make scale nicer < most plots seem to be close to 0 already
# note: there will be a very large peak at 0 for some of these plots 
# TODO: make it so that missing values aren't plotted at all
m,n = x_raw.shape
for row in range(0,m):
    for col in range(0,n):
        if x_raw[row][col] == -999.0:
             x_raw[row][col] = 0.0

for n in range(0,30):
    min_x_raw = min(x_raw[:,n])
    max_x_raw = max(x_raw[:,n])
    if min_x_raw < 0:
        binwidth = max_x_raw/num_bins 
    elif min_x_raw >= 0:
        binwidth = (max_x_raw-min_x_raw)/num_bins
    #plt.figure(n)
    plt.subplot(6,5,n+1)
    plt.hist(x_raw[:,n],
             bins=np.arange(min_x_raw, max_x_raw + binwidth, binwidth),range=[min_x_raw,max_x_raw])
    plt.title(header[n])
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    

plt.show()
'''
