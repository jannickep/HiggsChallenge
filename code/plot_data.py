# Author: Jannicke Pearkes
# Purpose: Plots the raw data from the Higgs Challenge

__docformat__ = 'restructedtext en'

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

def plot_data():
    # Read data
    with open("../data/training.csv","rb") as f:
        raw_data = list(csv.reader( f, delimiter=','))

    # Assign inputs (x_raw), outputs(y_raw) and weights (w_raw)
    header = list(raw_data[0])
    x_raw = np.array([map(float, row[1:-1]) for row in raw_data[1:]])
    y_raw = np.array([map(str, row[-1]) for row in raw_data[1:]])
    # Change to floats
    y_raw = np.where(y_raw == 's', 1.0, 0.0)
    
    # Put back together
    data = np.append(x_raw, y_raw, axis = 1)


    def split(arr, cond):
       return [arr[cond], arr[~cond]]

    [signal,background] = split(data, data[:,-1]==1.0)
    
    print "signal"
    print signal 
    print "background"
    print background
    num_bins = 30
    figure_num = 0


       # Make font size small
    mpl.rcParams.update({'font.size': 8})

    for col in range(0,30):
        [value, wrong_value] = split (data, data[:,col]!=-999.0)
        print "Number of -999s = "+str(wrong_value.size)
        [real_signal, real_background] = split(value, value[:,-1] == 1.0)
        print "real_background weights"
        print real_background[:,-2]
        print "real_signal weights"
        print real_signal[:,-2]
        if (int(col%6) == 0):
            plt.cla()
            plt.figure(figure_num)
            figure_num += 1 
        
        min_x_calc = min(value[:,col])
        max_x_calc = max(value[:,col])
        binwidth = (max_x_calc - min_x_calc) / num_bins
        print "min_x_calc: " + str(min_x_calc)
        print "max_x_calc: " + str(max_x_calc)
        print "binwidth: "+ str(binwidth)
        plt.subplot(2,3,(col%6)+1)
        plt.tight_layout()
        p1 = plt.hist([real_background[:,col], real_signal[:,col]],
                  #range=[min_x_calc,max_x_calc],
                  bins = np.arange(min_x_calc, max_x_calc + binwidth, binwidth),
                  weights = [real_background[:,-2],real_signal[:,-2]],
                  label = ['background','signal'],
                  linewidth = 0.0, 
                  edgecolor = None,
                  histtype = 'barstacked',
                  color = ['blue','red'],
                  alpha = 0.4 )
                  
        plt.legend(loc=1,prop={'size':6})
        plt.title(header[col+1])
        plt.xlabel("Value")
        plt.ylabel("Counts/Bin")
        #plt.text(0.0, 500, 'Number of invalid points:'+str(m-count[col]), fontsize=6)
        if (int(col%6) == 5):
            plt.savefig("../plots/weighted_raw_data_"+str(figure_num)+".png")

    plt.show()


if __name__ == '__main__':
    plot_data()
