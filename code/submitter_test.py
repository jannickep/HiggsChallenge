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

import theano
import theano.tensor as T

print '"submission" set is being loaded'
#test = list(csv.reader(open("../data/test.csv","rb"), delimiter=','))
#test_x = np.array([map(float, row[1:-2]) for row in test[1:5]])
#test_id = np.array([int(row[0]) for row in test[1:5]])
test_id = np.array([300,301,302,303])
test_x = np.array([1,1,1,1])
test_y = np.array([0,1,0,1])
test_z = np.array([[1,2],[3,4],[5,6],[7,8]])
nums_rows_test = 4
print 'test_x ' + repr(test_x)
print 'test_id' + repr(test_id)
print 'test_x shape: ' 
print test_x.shape
print 'test_id shape:  ' 
print test_id.shape
array = np.array(test_z[:,1])
print "array"
print repr(array)
order = array.argsort()
ranks = order.argsort()
print 'ranks'
print repr(ranks)
ranks = ranks +1
print repr(ranks)
submission = np.array([[str(test_id[tI]),str(ranks[tI]),'s' if test_y[tI] >= 1 else 'b'] for tI in range(len(test_id))])
#submission = np.array([[str(test_id[tI]),str(test_z[tI][1]),'s' if test_y[tI] >= 1 else 'b'] for tI in range(len(test_id))])
#submission = np.vstack((test_id, test_x, test_x))
print 'submission shape:  ' 
print submission.shape
print repr(submission)
#submission = np.append([['EventId','Rank','Class']], np.transpose(submission), axis=0)
print('submission is being saved')
np.savetxt("submission.csv",submission,fmt='%s',delimiter=',')
print('complete')


