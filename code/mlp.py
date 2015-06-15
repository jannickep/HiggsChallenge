"""
This tutorial introduces the multilayer perceptron using Theano.

 A multilayer perceptron is a logistic regressor where
instead of feeding the input to the logistic regression you insert a
intermediate layer, called the hidden layer, that has a nonlinear
activation function (usually tanh or sigmoid) . One can use many such
hidden layers making the architecture deep. The tutorial will also tackle
the problem of MNIST digit classification.

.. math::

    f(x) = G( b^{(2)} + W^{(2)}( s( b^{(1)} + W^{(1)} x))),

References:

    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 5

"""
__docformat__ = 'restructedtext en'
#hello

import os
import sys
import time
import cPickle
import numpy
import pdb
import theano
import theano.tensor as T
import random,string,math,csv
import scipy.linalg.blas
import numpy as np
import theano.tensor as T
from logistic_sgd import LogisticRegression, load_data
import getopt

#from sklearn import preprocessin

# start-snippet-1
class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        self.input = input
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4
            W = theano.shared(value=W_values, name='W', borrow=True)
        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)
        self.W = W
        self.b = b
        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]
    def outputer(self,x):
       return T.tanh(T.dot(x, self.W) + self.b)


# start-snippet-2
class MLP(object):
    """Multi-Layer Perceptron Class
    """

    def __init__(self, rng, input, n_in, n_hidden, n_out):
        self.hiddenLayer = HiddenLayer(
            rng=rng,
            input=input,
            n_in=n_in,
            n_out=n_hidden,
            activation=T.tanh
        )

        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_out
        )
        self.L1 = (
            abs(self.hiddenLayer.W).sum()
            + abs(self.logRegressionLayer.W).sum()
        )
        self.L2_sqr = (
            (self.hiddenLayer.W ** 2).sum()
            + (self.logRegressionLayer.W ** 2).sum()
        )
        self.negative_log_likelihood = (
            self.logRegressionLayer.negative_log_likelihood
        )
        self.errors = self.logRegressionLayer.errors
        self.params = self.hiddenLayer.params + self.logRegressionLayer.params
        self.layers = [self.hiddenLayer, self.logRegressionLayer] 
    def output(self, x):
        print "x"
        #print x.eval()
        y = self.hiddenLayer.outputer(x)
        print "y"
        #print y.eval()
        z = self.logRegressionLayer.output(y)
        print "z"
        #print z
        # Recursively compute output 
        #shared_x = T.matrix('shared_test_x')
        #shared_x = theano.shared(numpy.asarray(x, dtype=theano.config.floatX), borrow=True)
        #shared_x = theano.function((), self.hiddenLayer.output, givens={x: shared_x})
        #shared_x = theano.shared(numpy.asarray(x, dtype=theano.config.floatX), borrow=True)
        #pred = theano.function((), classifier.logRegressionLayer.p_y_given_x, givens={x: shared_x})
        return z


#def test_mlp(learning_rate=0.998, L1_reg=0.00005, L2_reg=0.000005,n_epochs=10000, 
#             batch_size, patience, patience_increase, improvement_threshold,
#             n_hidden, submit_threshold):
def test_mlp(learning_rate,
                  L1_reg,
                  L2_reg,
                  n_hidden,
                  n_epochs,
                  batch_size,
                  patience,
                  patience_increase,
                  improvement_threshold,
                  submit_threshold):
    """
    Demonstrate stochastic gradient descent optimization for a multilayer
    perceptron

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
    gradient

    :type L1_reg: float
    :param L1_reg: L1-norm's weight when added to the cost (see
    regularization)

    :type L2_reg: float
    :param L2_reg: L2-norm's weight when added to the cost (see
    regularization)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: the path of the MNIST dataset file from
                 http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz

    n_hidden original 500 (slightly less than inputs) maybe try 26 for 30 inputs?
   """

  
    datasets = load_data()
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
    print '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels
    #TODO: change 1234 to different number?
    rng = numpy.random.RandomState(1234)
    # construct the MLP class
    classifier = MLP(
        rng=rng,
        input=x,
        n_in=30,
        n_hidden=n_hidden,
        n_out=2
    )

    # start-snippet-4
    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = (
        classifier.negative_log_likelihood(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )
    # end-snippet-4

    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    # start-snippet-5
    # compute the gradient of cost with respect to theta (sotred in params)
    # the resulting gradients will be stored in a list gparams
    gparams = [T.grad(cost, param) for param in classifier.params]

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs

    # given two list the zip A = [a1, a2, a3, a4] and B = [b1, b2, b3, b4] of
    # same length, zip generates a list C of same size, where each element
    # is a pair formed from the two lists :
    #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]

    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # end-snippet-5

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'

    #TODO: Try changing there parameters around a bit for tuning 
    # early-stopping parameters
    #patience = 10000  # look as this many examples regardless original 10000
    #patience_increase = 2  # wait this much longer when a new best is
                           # found
    #improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant original .995
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if (
                        this_validation_loss < best_validation_loss *
                        improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [test_model(i) for i
                                   in xrange(n_test_batches)]
                    test_score = numpy.mean(test_losses)
                    test_std_dev = numpy.std(test_losses)

                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
    parameters = dict(learning_rate = learning_rate,
                  L1_reg = L1_reg,
                  L2_reg = L2_reg,
                  n_hidden = n_hidden,
                  n_epochs = n_epochs,
                  batch_size = batch_size,
                  patience = patience,
                  patience_increase = patience_increase,
                  improvement_threshold = improvement_threshold,
                  submit_threshold = submit_threshold
    ) 
    for key,value in parameters.items():
        print key,value
    
    print ("Values: ~["),
    for key,value in parameters.items():
        print (key+","),
    print ("]~")

    print ("Matrix: {["),
    for key,value in parameters.items():
        print (str(value)+","),
    print (str(test_score*100)+","+str(test_std_dev*100)+","
           +str((end_time-start_time)/60.0)+"]}")

    '''
    print ("Matrix: {["+str(learning_rate)+","+str(n_epochs)+","
           +str(batch_size)+","+str(patience)+","+str(patience_increase)+","
           +str(improvement_threshold)+","+str(submit_threshold)+","
           +str(test_score*100)+","+str(test_std_dev*100)+","
           +str((end_time-start_time)/60.0)+"]}")
    '''
    ######################
    # COMPUTE SUBMISSION #
    ######################


    arg1 = 0
    if (arg1== 1): 
        print 'submission set is being loaded'
        with open("../data/test_data.p", 'rb') as f:
             test_x, test_id = cPickle.load(f)

        shared_test_x = T.matrix('shared_test_x')
        print('prediction is being calculated')
        mlp_output =theano.function([shared_test_x],classifier.output(shared_test_x))
        print mlp_output(test_x)
        test_pred = mlp_output(test_x)
        print "size test_pred" 
        print test_pred.shape
        print "test_pred"
        print test_pred
        test_prob = test_pred
        print "test_prob"
        print test_prob      
        print("ranking submission events")
        array = np.array(test_prob[:,1])
        order = array.argsort()
        ranks = order.argsort()
        ranks = ranks +1
        submission = np.array([[str(test_id[tI]),
                                str(ranks[tI]),
                                's' if test_pred[tI][1] > submit_threshold 
                                 else 'b'] for tI in range(len(test_id))])
        submission = np.append([['EventId','RankOrder','Class']], 
                                submission, axis=0)     
        print('submission is being saved')
        np.savetxt("submission.csv",submission,fmt='%s',delimiter=',')
        print('complete')


if __name__ == '__main__':
    # Created dictionary to make things "slightly" less complicated with this number of parameters
    if len(sys.argv)>1:
        print "Using given values"
        name,learning_rate,n_epochs,batch_size,patience,patience_increase,improvement_threshold,submit_threshold,n_hidden = sys.argv
        parameters = dict(learning_rate = learning_rate,
                  L1_reg = L1_reg,
                  L2_reg = L2_reg,
                  n_hidden = n_hidden,
                  n_epochs = n_epochs,
                  batch_size = batch_size,
                  patience = patience,
                  patience_increase = patience_increase,
                  improvement_threshold = improvement_threshold,
                  submit_threshold = submit_threshold
                  ) 
        test_mlp(**parameters)

    else:
        print "Using hard-coded values"
        parameters = dict(learning_rate = 0.998,
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
        test_mlp(**parameters)

        #Hello 
