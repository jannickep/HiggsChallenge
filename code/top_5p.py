#top_5p.py
# Author: Jannicke Pearkes
# Purpose: take only top 5% of data and save as output array
# note, must be run after output_parser.py

import cPickle
import gzip
import os
import sys
import time
import random,string,math,csv
import numpy
import numpy as np

array = cPickle.load(open("array.p",'rb'))

ordered = array[array[:,-2].argsort()]

length = len(ordered)

cPickle.dump( ordered[0:int(length*float(sys.argv[1]))], open( "array.p", "wb" ) )

