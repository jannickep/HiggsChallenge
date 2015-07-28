# Compilation of generic functions for use with all theano neural networks
# Author: Jannicke Pearkes 
# jpearkes@uvic.ca

import cPickle
import gzip
import os
import sys
import time
import random,string,math,csv
import numpy
import numpy as np
import theano
import theano.tensor as T
from random import shuffle
import matplotlib 
import matplotlib.pyplot as plt
from itertools import cycle

def load_data():

    #############
    # LOAD DATA #
    #############  

    def shared_dataset(data_xy, borrow=True):
        data_x, data_y, data_w = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,dtype=theano.config.floatX), borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,dtype=theano.config.floatX), borrow=borrow)
        shared_w = theano.shared(numpy.asarray(data_w,dtype=theano.config.floatX), borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32'), shared_w        
    
    print "loading data"

    if len(sys.argv)>1 and len(sys.argv)<3:
    	level = sys.argv[1] # low level = 0, high level = 1
    else:
    	print "WARNING: No parameters passed - defaulting to high level inputs"
    	level = '1'
    if level == '1' :
        f = open("../data/normalizedTraining_high.p", 'rb') # high level inputs
        print "Using high level inputs"
    elif level == '2' :
        f = open("../data/normalizedTraining_high_met.p", 'rb') # high level inputs
        print "Using high level inputs"
    elif level == '3' :
        f = open("../data/normalizedTraining_high_met_sumet.p", 'rb') # high level inputs
        print "Using high level inputs"
    elif level == '4' :
        f = open("../data/normalizedTraining_high_tower.p", 'rb') # high level inputs
        print "Using high level inputs"
    elif level == '5' :
        f = open("../data/normalizedTraining_high_no_npv.p", 'rb') # high level inputs
        print "Using high level inputs"
    else: 
    	f = open("../data/normalizedTraining_low.p", 'rb') # include low level inputs
    	print "Using low level inputs" 
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    # Format Data for Theano 
    #TODO: see if can cut down this section 
    x = np.append(train_set[0],valid_set[0])
    x = np.append(x, test_set[0])
    y = np.append(train_set[1],valid_set[1])
    y = np.append(y, test_set[1])
    w = np.append(train_set[2],valid_set[2])
    w = np.append(w, test_set[2])
 
    yw = np.vstack((y,w))
    ywn = yw.transpose()
    print "x_size"
    print x.size
    length = y.size
    print "y_size"
    print y.size
    print "width_x"
    width_x = x.size/length
    print width_x
    #x = np.reshape(x,(250000,30))
    x = np.reshape(x,(length,width_x))
    xyw = np.hstack((x,ywn))

    #xyw = np.reshape(xyw, (250000,32))
    xyw = np.reshape(xyw, (length,width_x+2))

    #np.random.shuffle(xyw) #(np.random.permutation(xyw.transpose())).transpose()
    
    #len = 250000
    #length = 40000
    #Split into training,validation and test datasets
    train_set = (xyw[0:int(0.8*length),0:width_x],xyw[0:int(0.8*length),-2],xyw[0:int(0.8*length),-1])
    test_set =  (xyw[int(0.8*length):int(0.9*length),0:width_x],xyw[int(0.8*length):int(0.9*length),-2],xyw[int(0.8*length):int(0.9*length),-1])
    valid_set = (xyw[int(0.9*length):length,0:width_x],xyw[int(0.9*length):length,-2],xyw[int(0.9*length):length,-1])

    print 'data is being converted into theano shared_dataset'
    test_set_x, test_set_y, test_set_w = shared_dataset(test_set)
    valid_set_x, valid_set_y, valid_set_w = shared_dataset(valid_set)
    train_set_x, train_set_y, train_set_w = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y, train_set_w), (valid_set_x, valid_set_y, valid_set_w),(test_set_x, test_set_y, test_set_w)]
    return rval, width_x


def load_data2(iteration):
    print "Using low level inputs iteration:"+str(iteration) 
    f = open("../data/low_data/normalizedTraining_low"+iteration+".p", 'rb') # include low level inputs
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    
    # Format Data for Theano 
    #TODO: see if can cut down this section 
    x = np.append(train_set[0],valid_set[0])
    x = np.append(x, test_set[0])
    y = np.append(train_set[1],valid_set[1])
    y = np.append(y, test_set[1])
    w = np.append(train_set[2],valid_set[2])
    w = np.append(w, test_set[2])
 
    yw = np.vstack((y,w))
    ywn = yw.transpose()
    print "x_size"
    print x.size
    length = y.size
    print "y_size"
    print y.size
    print "width_x"
    width_x = x.size/length
    print width_x
    #x = np.reshape(x,(250000,30))
    x = np.reshape(x,(length,width_x))
    xyw = np.hstack((x,ywn))

    #xyw = np.reshape(xyw, (250000,32))
    xyw = np.reshape(xyw, (length,width_x+2))

    #np.random.shuffle(xyw) #(np.random.permutation(xyw.transpose())).transpose()
    
    #len = 250000
    #length = 40000
    #Split into training,validation and test datasets
    train_set = (xyw[0:int(0.8*length),0:width_x],xyw[0:int(0.8*length),-2],xyw[0:int(0.8*length),-1])
    test_set =  (xyw[int(0.8*length):int(0.9*length),0:width_x],xyw[int(0.8*length):int(0.9*length),-2],xyw[int(0.8*length):int(0.9*length),-1])
    valid_set = (xyw[int(0.9*length):length,0:width_x],xyw[int(0.9*length):length,-2],xyw[int(0.9*length):length,-1])

    print 'data is being converted into theano shared_dataset'
    test_set_x, test_set_y, test_set_w = shared_dataset(test_set)
    valid_set_x, valid_set_y, valid_set_w = shared_dataset(valid_set)
    train_set_x, train_set_y, train_set_w = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y, train_set_w), (valid_set_x, valid_set_y, valid_set_w),(test_set_x, test_set_y, test_set_w)]
    return rval, width_x



def plot_improvement(measures, colour):
    n_events = measures[:,0]
    percent = 1-measures[:,1]
    S = measures[:,2]
    B = measures[:,3]
    s = measures[:,4]
    b = measures[:,5]
    s_sqrt_b = np.sqrt(np.divide(s,b))
    sig_eff = np.divide(s,S)
    bg_eff = np.divide(b,B)
    bg_reg = 1- bg_eff
    purity = np.divide(s,np.add(s,b))
    #print "measures:"
    #print measures
    print " n_events"
    print n_events
    print "percent"
    print percent
    plt.figure(0)
    plt.plot(n_events,percent,colour+'-',label = "Percent Accuracy (%)")
    #plt.plot(n_events,S,label = "S")
    #plt.plot(n_events,B,label = "B")
    #plt.plot(n_events,s,label = "s")
    #plt.plot(n_events,b,label = "b")
    #plt.plot(n_events,s_sqrt_b,label = "s/sqrt(b)")
    plt.plot(n_events,sig_eff,colour+'--',label = "signal efficiency (s/S)")
    #plt.plot(n_events,bg_eff,'r--',label = "background efficiency (b/B)")
    plt.plot(n_events,bg_reg,colour+'-.',label = "background rejection 1-  (b/B)")
    #plt.plot(n_events,purity,label = "purity (s/(s+b))")
    plt.title("Improvement with Increased Training")
    plt.xlabel("Number of Training Epochs")
    plt.ylabel(" ")
    plt.ylim((-0.1,1.1))
    plt.legend(loc='lower right', shadow=False, ncol=1)
    plt.savefig("improvement.pdf")
    

def plot_roc(measures):
    plt.figure(1)
    n_events = measures[:,0]
    percent = 1-measures[:,1]
    S = measures[:,2]
    B = measures[:,3]
    s = measures[:,4]
    b = measures[:,5]
    s_sqrt_b = np.sqrt(np.divide(s,b))
    sig_eff = np.divide(s,S)
    bg_eff = 1-np.divide(b,B)
    purity = np.divide(s,np.add(s,b))
    #print "measures:"
    #print measures
    print " n_events"
    print n_events
    print "percent"
    print percent
    print "sig_eff"
    print sig_eff
    print "bg_eff"
    print bg_eff
    plt.plot(sig_eff,bg_eff,label = "ROC Curve")
    plt.title("ROC Curve")
    plt.xlabel("Signal Efficiency")
    plt.ylabel("Background Rejection")
    plt.legend(loc='lower right', shadow=False, ncol=1)
    plt.savefig("roc.pdf")
    

# Runs an roc curve test
def roc_run():
    measures2 = np.array([]).reshape(0,6)
    for i in np.arange(0,1.2,0.2):
        parameters = dict(
            learning_rate = 0.13, 
            n_epochs = 1000,
            batch_size = 600,
            patience = 5000, 
            patience_increase = 2,
            improvement_threshold = 0.995,
            submit_threshold = i)
        last_measures = sgd_optimization(**parameters)
        measures2 = np.vstack((measures2,last_measures))
    plot_roc(measures2)
    '''
    for i in np.arange(0,1.2,0.2):
        parameters = dict(
            learning_rate = 0.13, 
            n_epochs = 1000,
            batch_size = 600,
            patience = 5000, 
            patience_increase = 2,
            improvement_threshold = 0.995,
            submit_threshold = i)
        last_measures = mlp_optimization(**parameters)
        measures2 = np.vstack((measures2,last_measures))
    plot_roc(measures2)
        for i in np.arange(0,1.2,0.2):
        parameters = dict(
            learning_rate = 0.13, 
            n_epochs = 1000,
            batch_size = 600,
            patience = 5000, 
            patience_increase = 2,
            improvement_threshold = 0.995,
            submit_threshold = i)
        last_measures = sda_optimization(**parameters)
        measures2 = np.vstack((measures2,last_measures))
    plot_roc(measures2)
    '''


if __name__ == '__main__':
	print "This program only contains generic functions for use with the networks"
