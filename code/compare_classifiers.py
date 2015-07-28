#Purpose: To compare signal efficiencies e.t.c for different classifiers


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
from my_utils import *#load_data
from logistic_sgd import*
from mlp import*
from SdA_batch import*


def compare_classifiers():
    parameters = dict(
        learning_rate = 0.13, 
        n_epochs = 1000,
        batch_size = 600,
        patience = 5000, 
        patience_increase = 2,
        improvement_threshold = 0.995,
        submit_threshold = 0.5
        )
    measures = sgd_optimization(**parameters)
    plot_improvement(measures,'r')
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
    measures = test_mlp(**parameters)
    plot_improvement(measures,'b')
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
    measures = test_SdA(**parameters)
    plot_improvement(measures,'g')

def call_classifier(name):
    if (name == "sgd"):
        parameters = dict(
        learning_rate = 0.13, 
        n_epochs = 1000,
        batch_size = 600,
        patience = 5000, 
        patience_increase = 2,
        improvement_threshold = 0.995,
        submit_threshold = 0.5
        )
        outputs = sgd_optimization(**parameters)
    elif (name == "mlp"):
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
        outputs = test_mlp(**parameters)
    elif (name == "sda"):
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
        outputs = test_SdA(**parameters)
    else: 
        print "Error: No classifier named"
        return
    return outputs

def calculate_over_thresh(outputs, threshold):
    p_y_given_x = outputs[:,0]
    y = outputs[:,1]
    y_pred = np.where(p_y_given_x>threshold, 1, 0)
    S = np.sum(np.equal(y,1))
    B = np.sum(np.equal(y,0))
    s = np.sum(np.logical_and(np.equal(y,1),np.equal(y_pred,1)))
    b = np.sum(np.logical_and(np.equal(y,0),np.equal(y_pred,1)))
    return S,B,s,b

def calculate_efficiencies(outputs, threshold):
    S,B,s,b = calculate_over_thresh(outputs,threshold)
    sig_eff = np.divide(s,float(S))
    bg_eff = np.divide(b,float(B))
    bg_rej = 1-bg_eff
    return sig_eff,bg_rej

def calculate_roc(outputs):
    # Create array containing calculated efficiencies over a range of thresholds
    efficiencies = np.array([]).reshape(0,2)
    for threshold in np.arange(0,1.1,0.1):
        efficiencies_i = calculate_efficiencies(outputs,threshold)
        print "efficiencies_i"
        print efficiencies_i
        efficiencies = np.vstack((efficiencies,efficiencies_i))
    print efficiencies
    return efficiencies

def plot_roc(efficiencies,tag,colour):
    plt.figure(1)
    sig_eff = efficiencies[:,0]
    bg_eff = efficiencies[:,1]
    print "sig_eff"
    print sig_eff
    print "bg_eff"
    print bg_eff
    plt.plot(sig_eff,bg_eff, colour, label = tag)
    plt.title("ROC Curves")
    plt.xlabel("Signal Efficiency")
    plt.ylabel("Background Rejection")
    plt.legend(loc='lower left', shadow=False, ncol=1)
    plt.savefig("roc"+sys.argv[1]+".pdf")

def plot_rocs():
    # Calls logistic regression and obtains prediction from classifier
    # and actual value on test data

    tag = "logistic regression"
    colour = 'r'
    outputs = call_classifier("sgd")
    efficiencies = calculate_roc(outputs)
    plot_roc(efficiencies, tag, colour)

    tag = "multlayer perceptron"
    colour = 'b'
    outputs = call_classifier("mlp")
    efficiencies = calculate_roc(outputs)
    plot_roc(efficiencies, tag, colour)
    tag = "stacked denoising auto-encoder"
    colour = 'g'
    outputs = call_classifier("sda")
    efficiencies = calculate_roc(outputs)
    plot_roc(efficiencies, tag, colour)

        # for all in threshold_list
        # calculate efficiencies given threshold, should i split this up into two functions??
        # return (sig_eff, bg_eff) and add to list
    # plot roc in given colour


if __name__ == '__main__':
    #compare_classifiers()
    plot_rocs()


