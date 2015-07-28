"""
 Stacked Denoising Autoencoder from the theano tutorials
 Modified to work with the Higgs Challenge data
 Modified by: Jannicke Pearkes
 This tutorial introduces stacked denoising auto-encoders (SdA) using Theano.


"""
import os
import sys
import time
import cPickle
import math
import numpy
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer
from dA import dA


# start-snippet-1
class SdA(object):
    """Stacked denoising auto-encoder class (SdA)

    A stacked denoising autoencoder model is obtained by stacking several
    dAs. The hidden layer of the dA at layer `i` becomes the input of
    the dA at layer `i+1`. The first layer dA gets as input the input of
    the SdA, and the hidden layer of the last dA represents the output.
    Note that after pretraining, the SdA is dealt with as a normal MLP,
    the dAs are only used to initialize the weights.
    """

    def __init__(
        self,
        numpy_rng,
        theano_rng=None,
        n_ins=5,
        hidden_layers_sizes=[28, 28],#original [500,500]
        n_outs=2,
        corruption_levels=[0.1, 0.1],
        discriminant_threshold = 0.5
    ):
        
        self.sigmoid_layers = []
        self.dA_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)

        assert self.n_layers > 0

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        # allocate symbolic variables for the data
        self.x = T.matrix('x')  # the data is presented as rasterized images
        self.y = T.ivector('y')  # the labels are presented as 1D vector of
                                 # [int] labels
        # end-snippet-1

        # The SdA is an MLP, for which all weights of intermediate layers
        # are shared with a different denoising autoencoders
        # We will first construct the SdA as a deep multilayer perceptron,
        # and when constructing each sigmoidal layer we also construct a
        # denoising autoencoder that shares weights with that layer
        # During pretraining we will train these autoencoders (which will
        # lead to chainging the weights of the MLP as well)
        # During finetunining we will finish training the SdA by doing
        # stochastich gradient descent on the MLP

        # start-snippet-2
        for i in xrange(self.n_layers):
            # construct the sigmoidal layer

            # the size of the input is either the number of hidden units of
            # the layer below or the input size if we are on the first layer
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layers_sizes[i - 1]

            # the input to this layer is either the activation of the hidden
            # layer below or the input of the SdA if you are on the first
            # layer
            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[-1].output

            sigmoid_layer = HiddenLayer(rng=numpy_rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=hidden_layers_sizes[i],
                                        activation=T.nnet.sigmoid)
            # add the layer to our list of layers
            self.sigmoid_layers.append(sigmoid_layer)
   
            self.params.extend(sigmoid_layer.params)

            # Construct a denoising autoencoder that shared weights with this
            # layer
            dA_layer = dA(numpy_rng=numpy_rng,
                          theano_rng=theano_rng,
                          input=layer_input,
                          n_visible=input_size,
                          n_hidden=hidden_layers_sizes[i],
                          W=sigmoid_layer.W,
                          bhid=sigmoid_layer.b)
            self.dA_layers.append(dA_layer)
        # end-snippet-2
        # We now need to add a logistic layer on top of the MLP
        self.logLayer = LogisticRegression(
            input=self.sigmoid_layers[-1].output,
            n_in=hidden_layers_sizes[-1],
            n_out=n_outs,
            discriminant_threshold = discriminant_threshold
        )

        self.params.extend(self.logLayer.params)
        # construct a function that implements one step of finetunining

        # compute the cost for second phase of training,
        # defined as the negative log likelihood
        self.finetune_cost = self.logLayer.negative_log_likelihood(self.y)
        self.errors = self.logLayer.errors(self.y)
        self.py_given_x = self.logLayer.p_y_given_x

    def output(self, x):
        x = self.sigmoid_layers[0].outputer(x)
        x = self.sigmoid_layers[1].outputer(x)
        x = self.logLayer.output(x)
        return x
    def asimov_errors(self, y):
        # check if y has same dimension of y_pred
        if y.ndim != self.logLayer.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            S = T.sum(T.eq(y,1))
            B = T.sum(T.eq(y,0))#*10000 # TODO: cross-section scaling
            s = T.sum(T.and_(T.eq(y,1),T.eq(self.logLayer.y_pred,1)))
            b = T.sum(T.and_(T.eq(y,0),T.eq(self.logLayer.y_pred,1)))#*10000 TODO: cross-section scaling
            return(S,B,s,b)
            # represents a mistake in prediction
        else:
            raise NotImplementedError()

    def prediction(self,y):
        p_y_and_y = self.logLayer.p_y_given_x[:,1], y
        return  p_y_and_y



    def pretraining_functions(self, train_set_x, batch_size):
        ''' Generates a list of functions, each of them implementing one
        step in trainnig the dA corresponding to the layer with same index.
        The function will require as input the minibatch index, and to train
        a dA you just need to iterate, calling the corresponding function on
        all minibatch indexes.

        :type train_set_x: theano.tensor.TensorType
        :param train_set_x: Shared variable that contains all datapoints used
                            for training the dA

        :type batch_size: int
        :param batch_size: size of a [mini]batch

        :type learning_rate: float
        :param learning_rate: learning rate used during training for any of
                              the dA layers
        '''

        # index to a [mini]batch
        index = T.lscalar('index')  # index to a minibatch
        corruption_level = T.scalar('corruption')  # % of corruption to use
        learning_rate = T.scalar('lr')  # learning rate to use
        # begining of a batch, given `index`
        batch_begin = index * batch_size
        # ending of a batch given `index`
        batch_end = batch_begin + batch_size

        pretrain_fns = []
        for dA in self.dA_layers:
            # get the cost and the updates list
            cost, updates = dA.get_cost_updates(corruption_level,
                                                learning_rate)
            # compile the theano function
            fn = theano.function(
                inputs=[
                    index,
                    theano.Param(corruption_level, default=0.2),
                    theano.Param(learning_rate, default=0.1)
                ],
                outputs=cost,
                updates=updates,
                givens={
                    self.x: train_set_x[batch_begin: batch_end]
                }
            )
            # append `fn` to the list of functions
            pretrain_fns.append(fn)

        return pretrain_fns

    def build_finetune_functions(self, datasets, batch_size, learning_rate):
        '''Generates a function `train` that implements one step of
        finetuning, a function `validate` that computes the error on
        a batch from the validation set, and a function `test` that
        computes the error on a batch from the testing set

        :type datasets: list of pairs of theano.tensor.TensorType
        :param datasets: It is a list that contain all the datasets;
                         the has to contain three pairs, `train`,
                         `valid`, `test` in this order, where each pair
                         is formed of two Theano variables, one for the
                         datapoints, the other for the labels

        :type batch_size: int
        :param batch_size: size of a minibatch

        :type learning_rate: float
        :param learning_rate: learning rate used during finetune stage
        '''

        (train_set_x, train_set_y, train_set_w) = datasets[0]
        (valid_set_x, valid_set_y, valid_set_w) = datasets[1]
        (test_set_x, test_set_y, test_set_w) = datasets[2]

        # compute number of minibatches for training, validation and testing
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
        n_valid_batches /= batch_size
        n_test_batches = test_set_x.get_value(borrow=True).shape[0]
        n_test_batches /= batch_size

        index = T.lscalar('index')  # index to a [mini]batch

        # compute the gradients with respect to the model parameters
        gparams = T.grad(self.finetune_cost, self.params)

        # compute list of fine-tuning updates
        updates = [
            (param, param - gparam * learning_rate)
            for param, gparam in zip(self.params, gparams)
        ]

        train_fn = theano.function(
            inputs=[index],
            outputs=self.finetune_cost,
            updates=updates,
            givens={
                self.x: train_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: train_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name='train'
        )

        test_score_i = theano.function(
            [index],
            self.errors,
            givens={
                self.x: test_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: test_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name='test'
        )

        test_asimov_model = theano.function(
            inputs=[index],
            outputs=self.asimov_errors(self.y),
            givens={
                self.x: test_set_x[index * batch_size: (index + 1) * batch_size],
                self.y: test_set_y[index * batch_size: (index + 1) * batch_size]
                #w: test_set_w[index * batch_size: (index + 1) * batch_size] #JP
            }
        )
        
        get_prediction_model = theano.function(
            inputs=[index],
            outputs=self.prediction(self.y),
            givens={
                self.x: test_set_x[index * batch_size: (index + 1) * batch_size],
                self.y: test_set_y[index * batch_size: (index + 1) * batch_size]
                #w: test_set_w[index * batch_size: (index + 1) * batch_size] #JP
            }
        )

        valid_score_i = theano.function(
            [index],
            self.errors,
            givens={
                self.x: valid_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: valid_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name='valid'
        )

        # Create a function that scans the entire validation set
        def valid_score():
            return [valid_score_i(i) for i in xrange(n_valid_batches)]

        # Create a function that scans the entire test set
        def test_score():
            return [test_score_i(i) for i in xrange(n_test_batches)]

        def asimov_score():
            return np.sum([test_asimov_model(i) 
                                   for i in xrange(n_test_batches)], axis = 0)
        def prediction_score():
            return np.vstack([np.reshape(np.ravel(get_prediction_model(i), order='F'),(-1,2))
                                   for i in xrange(n_test_batches)])
        return train_fn, valid_score, test_score, asimov_score, prediction_score


def test_SdA(finetune_lr, patience, 
             patience_increase,submit_threshold, 
             improvement_threshold, pretraining_epochs,
             pretrain_lr, training_epochs,
             batch_size,neurons_per_layer,number_of_layers):
    #def test_SdA(finetune_lr=0.1, pretraining_epochs=15,
    #             pretrain_lr=0.001, training_epochs=1000,
    #             dataset='mnist.pkl.gz', batch_size=1,neurons_per_layer=32,number_of_layers=3):    

    """
    Demonstrates how to train and test a stochastic denoising autoencoder.

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used in the finetune stage
    (factor for the stochastic gradient)

    :type pretraining_epochs: int
    :param pretraining_epochs: number of epoch to do pretraining

    :type pretrain_lr: float
    :param pretrain_lr: learning rate to be used during pre-training

    :type n_iter: int
    :param n_iter: maximal number of iterations ot run the optimizer

    :type dataset: string
    :param dataset: path the the pickled dataset

    """
    measures = np.array([]).reshape(0,6)
    datasets,width_x = load_data()

    train_set_x, train_set_y, train_set_w = datasets[0]
    valid_set_x, valid_set_y, valid_set_w = datasets[1]
    test_set_x, test_set_y, test_set_w = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size

    # numpy random generator
    # start-snippet-3
    numpy_rng = numpy.random.RandomState(89677)
    print '... building the model'
    # construct the stacked denoising autoencoder class
    sda = SdA(
        numpy_rng=numpy_rng,
        n_ins=width_x,
        #hidden_layers_sizes=[32, 32, 32], # orginally [1000, 1000, 1000] 
        hidden_layers_sizes=np.ones((number_of_layers,), dtype=int)*neurons_per_layer,
        n_outs=2, discriminant_threshold = submit_threshold
    )

    #########################
    # PRETRAINING THE MODEL #
    #########################
    print '... getting the pretraining functions'
    pretraining_fns = sda.pretraining_functions(train_set_x=train_set_x,
                                                batch_size=batch_size)

    print '... pre-training the model'
    start_time = time.clock()
    ## Pre-train layer-wise
    corruption_levels = [.1, .2, .3, .3, .3] # not sure if these levels are ok
                                             # could not find good online source
    for i in xrange(sda.n_layers):
        # go through pretraining epochs
        for epoch in xrange(pretraining_epochs):
            # go through the training set
            c = []
            for batch_index in xrange(n_train_batches):
                c.append(pretraining_fns[i](index=batch_index,
                         corruption=corruption_levels[i],
                         lr=pretrain_lr))
            print 'Pre-training layer %i, epoch %d, cost ' % (i, epoch),
            print numpy.mean(c)

    end_time = time.clock()

    print >> sys.stderr, ('The pretraining code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

    ########################
    # FINETUNING THE MODEL #
    ########################

    # get the training, validation and testing function for the model
    print '... getting the finetuning functions'
    train_fn, validate_model, test_model, asimov_model, prediction_score = sda.build_finetune_functions(
        datasets=datasets,
        batch_size=batch_size,
        learning_rate=finetune_lr
    )

    print '... finetunning the model'
    # early-stopping parameters
    #patience = 10 # * n_train_batches  # look as this many examples regardless
    #patience_increase = 2.  # wait this much longer when a new best is
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

    while (epoch < training_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_fn(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                validation_losses = validate_model()
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if (
                        this_validation_loss < best_validation_loss *
                        improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = test_model()
                    p_y_and_y = prediction_score()
                    test_score = numpy.mean(test_losses)
                    #test_std_dev = numpy.std(test_losses)
                    test_std_dev = numpy.std(test_losses)/math.sqrt(len(test_losses))
                    output= asimov_model()
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

                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print(
        (
            'Optimization complete with best validation score of %f %%, '
            'on iteration %i, '
            'with test performance %f %%'
        )
        % (best_validation_loss * 100., best_iter + 1, test_score * 100.)
    )
    print >> sys.stderr, ('The training code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
    
    parameters = dict(
             improvement_threshold = improvement_threshold,
             finetune_lr = finetune_lr,
             patience = patience,
             patience_increase = patience_increase,
             pretraining_epochs = pretraining_epochs,
             pretrain_lr = pretrain_lr, 
             training_epochs = training_epochs,
             batch_size = batch_size,
             neurons_per_layer = neurons_per_layer,
             number_of_layers = number_of_layers, 
             submit_threshold = submit_threshold
             )

    for key,value in parameters.items():
        print key,value
    
    print ("Values: ~["),
    for key,value in parameters.items():
        print (key+","),
    print ("test_score, test_std_dev, walltime]~")

    print ("Matrix: {["),
    for key,value in parameters.items():
        print (str(value)+","),
    print (str(test_score*100)+","+str(test_std_dev*100)+","
           +str((end_time-start_time)/60.0)+"]}")


    ######################
    # COMPUTE SUBMISSION #
    ######################
    arg1 = 0
    if (arg1== 1):

        print '"submission" set is being loaded'
        f = open("test_data.p", 'rb')
        test_x, test_id = cPickle.load(f)
        f.close()

        shared_test_x = T.matrix('shared_test_x')
        print('prediction is being calculated')
        mlp_output =theano.function([shared_test_x], sda.output(shared_test_x))
        print mlp_output(test_x)
        test_pred = mlp_output(test_x)
        print "size test_pred"
        print test_pred.shape
        print "test_pred"
        print test_pred
        test_prob = test_pred
        print "test_prob"
        print test_prob
        #test_pred = theano.function((), classifier.logRegressionLayer.y_pred, givens={x: shared_test_x})
        #test_prob = theano.function((), classifier.logRegressionLayer.p_y_given_x, givens={x: shared_test_x})

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

    ''' 
    if (true):

        print '"submission" set is being loaded'
        test = list(csv.reader(open("../data/test.csv","rb"), delimiter=','))
        test_x = np.array([map(str, row[1:]) for row in test[1:]])
        test_id = np.array([int(row[0]) for row in test[1:]])
        nums_rows_test = len(test_x)

        shared_test_x = theano.shared(numpy.asarray(test_x, dtype=theano.config.floatX), borrow=True)
        print('prediction is being calculated')
        test_pred = theano.function((), classifier.logRegressionLayer.y_pred, givens={x: shared_test_x})
        test_prob = theano.function((), classifier.logRegressionLayer.p_y_given_x, givens={x: shared_test_x})
        test_pred_val = test_pred() # don't need to do this
        test_prob_val = test_prob() # don't need to do this

        array = np.array(test_prob_val[:,1])
        order = array.argsort()
        ranks = order.argsort()
        ranks = ranks +1
        submission = np.array([[str(test_id[tI]),
                                str(ranks[tI]),
                                's' if test_pred_val[tI] > 0.5 else 'b'] for tI in range(len(test_id))])
        submission = np.append([['EventId','RankOrder','Class']], submission, axis=0)
        print('submission is being saved')
        np.savetxt("submission.csv",submission,fmt='%s',delimiter=',')

    print('complete')

    '''
    #return measures
    return p_y_and_y

if __name__ == '__main__':
     
    if len(sys.argv)>2:
        print "Using passed parameters"
        name,batch_size,finetune_lr,improvement_threshold, neurons_per_layer,number_of_layers, patience, patience_increase, pretraining_epochs,pretrain_lr, submit_threshold,training_epochs = sys.argv
        parameters = dict(
                 improvement_threshold = float(improvement_threshold),
                 finetune_lr = float(finetune_lr),
                 patience = int(float(patience)),
                 patience_increase = int (float(patience_increase)),
                 pretraining_epochs = int(float(pretraining_epochs)),
                 pretrain_lr = float(pretrain_lr), 
                 training_epochs = int(float(training_epochs)),
                 batch_size = int(float(batch_size)),
                 neurons_per_layer = int(float(neurons_per_layer)),
                 number_of_layers = int(float(number_of_layers)), 
                 submit_threshold = float(submit_threshold)
                 )
        print parameters
        test_SdA(**parameters)
        

    else:
        print "Using hard-coded parameters"
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
        test_SdA(**parameters)

