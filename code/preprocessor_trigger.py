# Author: Jannicke Pearkes
# Purpose: Normalizes and shuffles training data for the Trigger Project

import os
import sys
import time
import random
import string
import math
import csv
import scipy.linalg.blas
import numpy as np
import cPickle
import pprint
import glob 

def pre_process_data():
    #Percentage for Training, Validation and Test Sets
    train_p = .80
    valid_p = .90
    test_p = 1
    iteration = 0

    bg_files = glob.glob("../data/low_data/bg_*.csv")
    sig_files = glob.glob("../data/low_data/sig_*.csv")
    
    for file_i in range(len(sig_files)):
        # Read in Data Set
        print "Reading training set ..."
        with open(sig_files[file_i],"rb") as f:
            sig_raw_data = list(csv.reader( f, delimiter=',')) 
        with open(bg_files[file_i],"rb") as f:
            bg_raw_data = list(csv.reader( f, delimiter=','))    
        header = list(sig_raw_data[0])

        # Slice off header
        sig_raw_data = sig_raw_data[1:]
        bg_raw_data = bg_raw_data[1:]

        # Initialize array
        len_sig = len(sig_raw_data)
        len_bg = len(bg_raw_data)

        # Make sure both data sets are the same size
        if len_sig != len_bg:
            print "Datasets are not the same size, splitting"
            if len_sig < len_bg:
                bg_raw_data = bg_raw_data[0:len_sig]
            else: 
                sig_raw_data = sig_raw_data[0:len_bg]

        raw_data = [0]*len(sig_raw_data*2)

        # Merge signal and background data sets
        raw_data[::2]= sig_raw_data
        raw_data[1::2]= bg_raw_data
        #---------------------------------------------------------
        for low,high in [(0,5000),(5000,10000)]:
            # Assign inputs (x_raw), outputs(y_raw) and weights (w_raw)
            #x_raw = np.array([map(float, row[1:-2]) for row in raw_data[:]])
            #x_raw = np.array([map(float, row[5:-2]) for row in raw_data[:]])
            x_raw = np.array([map(float, row[6:-2]) for row in raw_data[low:high]])
            #x_raw = np.array([map(float, row[0:2]) for row in raw_data[:]])
            #x_raw = np.array([map(float, row[0:5]) for row in raw_data[:]])
            #x_raw = np.delete(x_raw,2,axis = 1)
            #x_raw = np.array([map(float, row[3:5]) for row in raw_data[:]])
            #if type(raw_data[1][-1]) is str:
            #y_raw = np.array([map(str, row[-1]) for row in raw_data[:]])
            #else:
            y_raw = np.array([map(int, row[-1]) for row in raw_data[low:high]])   
            w_raw = np.array([float(row[-2]) for row in raw_data[low:high]])

            # Set 's' inputs to 1 and 'b' inputs to 0
            # y_raw = np.where(y_raw == 's', 1,0).flatten()
            # w_raw = w_raw.flatten()
            y_raw = y_raw.flatten()
            w_raw = w_raw.flatten()
            print "y_raw:"
            print y_raw

            # Normalize inputs
            print "Normalizing data ..."
            m,n = x_raw.shape
            print 'x_raw shape:'
            print  x_raw.shape

            #Initialize Arrays to 0
            sum_x = np.zeros(n) 
            count = np.zeros(n)
            mean = np.zeros(n)
            std_dev = np.zeros(n)
            x_new = np.zeros((m,n))
            x_calc = np.zeros((m,n))
            '''
            # Input valid data (not -999) into array
            for col in range(0,n):
                for row in range(0,m):
                    if (x_raw[row][col] != -9.99000000e+02):
                         x_calc[count[col]][col] = x_raw[row][col]
                         count[col] = count[col]+1 #keep track of number of valid data points

            #print 'x_calc:' 
            #print repr(x_calc)
            #print repr(count)
            #print 'x_raw:' 
            #print repr(x_raw)

            # Calculate mean and standard deviation for valid data
            for i in range (0,n):
                mean[i] = np.mean(x_calc[0:count[i],i])
                std_dev[i] = np.std(x_calc[0:count[i],i])
            print 'mean'
            print repr(mean)
            print 'std_dev'
            print repr(std_dev)
            # looks good to here
            # Use mean and standard deviation to normalize data
            for col in range(0,n):
                for row in range(0,m):
                    if ((x_raw[row][col] != -9.99000000e+02) and (std_dev[col]!= 0)): # std_dev being zero would create a nan
                        x_new[row][col] = (x_raw[row][col] - mean[col]) / std_dev[col] 
            '''    
            x_new = x_raw
            print "X_NEW:" 
            print repr(x_new)
            print "Y_NEW:"
            print y_raw
            print "W_NEW:"
            print w_raw
            

            # Create tuples for use with theano
            print "Creating tuples"
            train_set_x = x_new[0:int(m*train_p)]
            valid_set_x = x_new[int(m*train_p):int(m*valid_p)]
            test_set_x = x_new[int(m*valid_p):int(m*test_p)]

            train_set_y = y_raw[0:int(m*train_p)]
            valid_set_y = y_raw[int(m*train_p):int(m*valid_p)]
            test_set_y = y_raw[int(m*valid_p):int(m*test_p)]

            train_set_w = w_raw[0:int(m*train_p)]
            valid_set_w = w_raw[int(m*train_p):int(m*valid_p)]
            test_set_w = w_raw[int(m*valid_p):int(m*test_p)]

            train_set = train_set_x, train_set_y, train_set_w 
            valid_set = valid_set_x, valid_set_y, valid_set_w
            test_set = test_set_x, test_set_y, test_set_w

            data = train_set, valid_set, test_set
            print "data:"
            #print data
            print "Saving data at ../data/normalizedTraining_low_"+str(iteration)+".p"
            with  open("../data/normalizedTraining_low_"+str(iteration)+".p", "wb") as f:
                cPickle.dump( data, f )
            iteration += 1
        print "Preprocessing of training set complete"

if __name__ == '__main__':
    pre_process_data()
