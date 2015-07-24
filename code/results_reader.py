# Author: Jannicke Pearkes
# Purpose: To extract some simple information about the grid search
#          - mean
#          - std_dev
#          - best areas 
#          - worst areas 

# Author: Jannicke Pearkes
# Purpose: To evaluate the relative change in hyperparameters according to performance

import numpy as np
import cPickle
import logging


def get_index(header, string):
    logging.debug("header length: " +str(len(header)))
    for i in range(len(header)):
        logging.debug("header["+str(i)+"]"+header[i])
        if string in header[i] :
            return i



array = cPickle.load(open("array.p",'rb'))
header = cPickle.load(open("header.p",'rb'))
print header

#array = cPickle.load(open("linear_search.p",'rb'))
total = len(array)
print total

best_score = min(array[:,get_index(header,"test_score")])
print ("best_score: "+str(best_score))

y = array[:,get_index(header,"test_score")]

miny = np.argwhere(y == np.amin(y))
print ("min y:" + str( miny)+" at "+str(np.amin(y)))
for i in range(len(miny)):
    print ("x at min" + str(array[miny[i],:]))

maxy = np.argwhere(y == np.amax(y))
print ("max y:" + str( maxy)+" at "+str(np.amax(y)))
for i in range(len(maxy)):
    print ("x at min" + str(array[maxy[i],:]))

av_std_dev = np.mean(array[:,get_index(header,"test_std_dev")])
print "av_std_dev: "+str(av_std_dev)

low_std_dev = np.min(array[:,get_index(header,"test_std_dev")])
print "lowest_std_dev: "+str(low_std_dev)

hi_std_dev = np.max(array[:,get_index(header,"test_std_dev")])
print "highest_std_dev: "+str(hi_std_dev)

ordered = array[array[:,-3].argsort()]
print ordered[:,-3]
print ordered[:,-2]



