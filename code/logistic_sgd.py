"""
Theano Tutorial: Logistic Regression
Modified to work with the Kaggle Higgs Challenge
Modified by: Jannicke Pearkes, jpearkes@uvic.ca
"""
__docformat__ = 'restructedtext en'

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

class LogisticRegression(object):
    def __init__(self, input, n_in, n_out, discriminant_threshold):

        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
              
        # initialize the basis b as a vector of n_out 0s
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )
        
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        # edited to reject events below a threshold
        #self.y_pred = T.argmax(self.p_y_given_x, axis=1) 
        #self.y_pred = T.and_(T.argmax(self.p_y_given_x, axis=1), T.ge(self.p_y_given_x[:,1], -1))#discriminant_threshold))
        self.y_pred = T.ge(self.p_y_given_x[:,1], discriminant_threshold)#discriminant_threshold))

        # parameters of the model
        self.params = [self.W, self.b]
    
    #JP - Added output function
    def output(self,x):
        return  T.nnet.softmax(T.dot(x, self.W) + self.b)

    def pred(self,x):
        return  self.y_pred


    def negative_log_likelihood(self, y):
        #  the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])


    def errors(self, y):
        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            return T.mean(T.neq(self.y_pred, y))
            # represents a mistake in prediction
        else:
            raise NotImplementedError()


    def asimov_errors(self, y):
        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            S = T.sum(T.eq(y,1))
            B = T.sum(T.eq(y,0))#*10000 # TODO: cross-section scaling
            s = T.sum(T.and_(T.eq(y,1),T.eq(self.y_pred,1)))
            b = T.sum(T.and_(T.eq(y,0),T.eq(self.y_pred,1)))#*10000 TODO: cross-section scaling
            return(S,B,s,b)
            # represents a mistake in prediction
        else:
            raise NotImplementedError()

    def prediction(self,y):
        p_y_and_y = self.p_y_given_x[:,1], y
        return  p_y_and_y

'''
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
    f = open("../data/normalizedTraining_high.p", 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()


    # Format Data for Theano
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
    len_y = y.size
    print "y_size"
    print y.size
    print "width_x"
    width_x = x.size/len_y
    print width_x
    #x = np.reshape(x,(250000,30))
    x = np.reshape(x,(len_y,width_x))
    xyw = np.hstack((x,ywn))

    #xyw = np.reshape(xyw, (250000,32))
    xyw = np.reshape(xyw, (len_y,width_x+2))

    #np.random.shuffle(xyw) #(np.random.permutation(xyw.transpose())).transpose()
    
    #len = 250000
    len = 40000
    train_set = (xyw[0:int(0.8*len),0:width_x],xyw[0:int(0.8*len),-2],xyw[0:int(0.8*len),-1])
    test_set =  (xyw[int(0.8*len):int(0.9*len),0:width_x],xyw[int(0.8*len):int(0.9*len),-2],xyw[int(0.8*len):int(0.9*len),-1])
    valid_set = (xyw[int(0.9*len):len,0:width_x],xyw[int(0.9*len):len,-2],xyw[int(0.9*len):len,-1])

    print 'data is being converted into theano shared_dataset'
    test_set_x, test_set_y, test_set_w = shared_dataset(test_set)
    valid_set_x, valid_set_y, valid_set_w = shared_dataset(valid_set)
    train_set_x, train_set_y, train_set_w = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y, train_set_w), (valid_set_x, valid_set_y, valid_set_w),(test_set_x, test_set_y, test_set_w)]
    return rval
'''

 

def sgd_optimization(learning_rate, n_epochs,
                           batch_size, patience, 
                           patience_increase, improvement_threshold,
                           submit_threshold):
    measures = np.array([]).reshape(0,6)
    datasets,width_x = load_data()
    
    print 'finished loading data'
    # unpack datasets
    train_set_x, train_set_y, train_set_w = datasets[0]
    valid_set_x, valid_set_y, valid_set_w = datasets[1]
    test_set_x, test_set_y, test_set_w = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    #print train_set_x.size.eval()
    #width_x= train_set_x.size.eval()

    #width_x =5 #Change this
    print '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # generate symbolic variables for input (x and y represent a
    # minibatch)
    x = T.matrix('x')  # data, presented as rasterized images
    y = T.ivector('y')  # labels, presented as 1D vector of [int] labels
    w = T.ivector('w') # JP

    # construct the logistic regression class
    # Each set of data has size 30 
    # Number of outputs is 2 
    classifier = LogisticRegression(input=x, n_in=width_x, n_out=2, discriminant_threshold = submit_threshold)
    #classifier = LogisticRegression(input=x, n_in=28*28, n_out=10)

    # the cost we minimize during training is the negative log likelihood of
    # the model in symbolic format
    #cost = classifier.negative_log_likelihood(y)
    cost = classifier.negative_log_likelihood(y) #JP 
    
    # compiling a Theano function that computes the mistakes that are made by
    # the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
            #w: test_set_w[index * batch_size: (index + 1) * batch_size] #JP
        }
    )

    test_asimov_model = theano.function(
        inputs=[index],
        outputs=classifier.asimov_errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
            #w: test_set_w[index * batch_size: (index + 1) * batch_size] #JP
        }
    )

    get_prediction_model = theano.function(
        inputs=[index],
        outputs=classifier.prediction(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
            #w: test_set_w[index * batch_size: (index + 1) * batch_size] #JP
        }
    )

    output_model = theano.function(
        inputs=[index],
        outputs=classifier.pred(x),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            #y: test_set_y[index * batch_size: (index + 1) * batch_size]
            #w: test_set_w[index * batch_size: (index + 1) * batch_size] #JP
        }
    )
    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
            #,w: valid_set_w[index * batch_size: (index + 1) * batch_size] #JP
        }
    )

    # compute the gradient of cost with respect to theta = (W,b)
    g_W = T.grad(cost=cost, wrt=classifier.W)
    g_b = T.grad(cost=cost, wrt=classifier.b)

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs.
    updates = [(classifier.W, classifier.W - learning_rate * g_W),
               (classifier.b, classifier.b - learning_rate * g_b)]

    # compiling a Theano function `train_model` that returns the cost, but in
    # the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size],
           # w: train_set_w[index * batch_size: (index + 1) * batch_size]
        }
    )


    ###############
    # TRAIN MODEL #
    ###############
    print '... training the model'
    # early-stopping parameters
    #patience = 5000  # look as this many examples regardless
    #patience_increase = 2  # wait this much longer when a new best is
                                  # found
    #improvement_threshold = 0.995  # a relative improvement of this much is
                                  # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatches before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = time.clock()

    done_looping = False
    epoch = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i)
                                     for i in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                     (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    ))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    # test it on the test set

                    test_losses = [test_model(i)
                                   for i in xrange(n_test_batches)]
                    y_pred = np.array([output_model(i)
                                   for i in xrange(n_test_batches)])
                    p_y_and_y = np.vstack([np.reshape(np.ravel(get_prediction_model(i), order='F'),(-1,2))
                                   for i in xrange(n_test_batches)])
                    #p_y_and_y = np.reshape(p_y_and_y, (-1,2))
                    ### think problem is here
                    test_score = numpy.mean(test_losses)
                    #-----------------
                    #test_std_dev = numpy.std(test_losses)
                    test_std_dev = numpy.std(test_losses)/math.sqrt(len(test_losses))
                    output= np.sum([test_asimov_model(i) 
                                   for i in xrange(n_test_batches)], axis = 0)
                    S,B,s,b= output
                    S = float(S)
                    B = float(B)
                    s = float(s)
                    b = float(b)
                    print "S: "+str(S)
                    print "B: "+str(B)
                    print "s: "+str(s)
                    print "b: "+str(b)
                    #asimov_sig = (s/math.sqrt(b)) #approximation 
                    if(b!=0):
                        asimov_sig = math.sqrt(2*((s+b)*math.log(1+s/b)-s))#math.sqrt(2(s+b))#*math.log(1+s/b)-s))
                    else:
                        asimov_sig = 10000
                    print "asimov: "+str(asimov_sig)
                    n_events = epoch
                    measures = np.vstack((measures,np.array([n_events,test_score, S,B,s,b])))
                    #-----------------------
                    print(('     epoch %i, minibatch %i/%i, test error of'
                           ' best model %f %%') %
                        (
                            epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            test_score * 100.
                        )
                    )

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print(('Optimization complete with best validation score of %f %%,'
            'with test performance %f %%')
            % (best_validation_loss * 100., test_score * 100.))
    print "S: "+str(S)
    print "B: "+str(B)
    print "s: "+str(s)
    print "b: "+str(b)
    print "asimov: "+str(asimov_sig)
    print "y_pred:"
    print y_pred.flatten()
    print "MET"
    met = np.transpose(test_set_x[:,0].eval())
    print met 
    results = (met, test_losses)
    last_measures = np.array([submit_threshold,test_score, S,B,s,b])
    p_y_and_y 
    #plot_improvement(measures)

    print (('The code ran for %d epochs, with %f epochs/sec') % 
          (epoch, 1. * epoch / (end_time - start_time)))

    print "learning rate: " + str(learning_rate)
    print "epochs: " + str(n_epochs)
    print "batch size: " + str(batch_size)
    print "patience: " + str(patience)
    print "patience_increase: " + str(patience_increase)
    print "improvement_threshold" + str(improvement_threshold)
    print "submit_threshold" + str(submit_threshold)
    print ("Values: ~[learning_rate,n_epochs,batch_size,patience,"
           +"patience_increase,improvement_threshold,"
           +"submit_threshold,test_score,test_std_dev,time]~")
    print ("Matrix: {["+str(learning_rate)+","+str(n_epochs)+","
           +str(batch_size)+","+str(patience)+","+str(patience_increase)+","
           +str(improvement_threshold)+","+str(submit_threshold)+","
           +str(test_score*100)+","+str(test_std_dev*100)+","
           +str((end_time-start_time)/60.0)+"]}")

 
    ######################
    # COMPUTE SUBMISSION #
    ######################
    # TODO: would be really nice to have this as a function like load_data
    #       input = name of classifier
    arg1 = 0
    if (arg1== 1): 
        print 'submission set is being loaded'
        with open("../data/test_data.p", 'rb') as f:
             test_x, test_id = cPickle.load(f)
        nums_rows_test = len(test_x)
        shared_test_x = theano.shared(numpy.asarray(test_x, dtype=theano.config.floatX), borrow=True)
        print('prediction is being calculated')
        test_pred = theano.function((), classifier.y_pred, givens={x: shared_test_x})
        test_prob = theano.function((), classifier.p_y_given_x, givens={x: shared_test_x})
        test_pred_val = test_pred() # don't need to do this
        test_prob_val = test_prob() # don't need to do this
        array = np.array(test_prob_val[:,1])
        order = array.argsort()
        ranks = order.argsort() + 1 #ranking needs to start at 1
        submission = np.array([[str(test_id[tI]),
                                str(ranks[tI]),
                                's' if test_pred_val[tI] > submit_threshold else 'b'] 
                                for tI in range(len(test_id))])                        
        submission = np.append([['EventId','RankOrder','Class']], submission, axis=0)
        print('submission is being saved')
        np.savetxt("submission.csv",submission,fmt='%s',delimiter=',')
        print('complete')
    #return last_measures
    #return results
    #return measures
    print "p_y_and_y"
    print p_y_and_y
    return p_y_and_y

'''
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


if __name__ == '__main__':
    if len(sys.argv)>2:
        print "Using passed values"
        name,batch_size,improvement_threshold,learning_rate,n_epochs,patience,patience_increase,submit_threshold = sys.argv
        parameters = dict(
            learning_rate = float(learning_rate), 
            n_epochs = int(float(n_epochs)),
            batch_size =  int(float(batch_size)), 
            patience = int(float(patience)),
            patience_increase = int(float(patience_increase)),
            improvement_threshold = float(improvement_threshold),
            submit_threshold =  float(submit_threshold))
        sgd_optimization(**parameters)
        
    else:
        #roc_run()
        #turn_on_curve()
        
        print "Using hard-coded values"
        parameters = dict(
            learning_rate = 0.13, 
            n_epochs = 1000,
            batch_size = 600,
            patience = 5000, 
            patience_increase = 2,
            improvement_threshold = 0.995,
            submit_threshold = 0.5)
        sgd_optimization(**parameters)
        
