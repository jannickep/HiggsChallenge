# Author: Jannicke Pearkes
# Purpose: Parses the output files from the batch scheduler for results which are compiled into an array


import re,sys,glob
import numpy as np
import cPickle
import math
import logging

#logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(message)s')
logging.basicConfig(level=logging.CRITICAL, format='%(asctime)s - %(message)s')
# What files to look for
#files = glob.glob('mlp_optimization.o2919*')
if len(sys.argv)>1:
    files = glob.glob(sys.argv[1]+"*")
    print "using passed parameter"
else:
    files = glob.glob('linear_search.o293186*')
    print "no parameter specified"

rgx = re.compile(r"\{\[(.*?)\]\}") #TODO: Add in expression
rgx2 = re.compile(r"kb,walltime\=(.*)")
rgx3 = re.compile("\~\[(.*?)\]\~") #TODO: Add in expression
array = np.zeros((1,15))
count = 0
save_word = None
save_time = None
save_header = ""
file_count = 0
good_file_count = 0

def get_sec(s):
    l = s.split(':')
    return (float(l[0]) * 60 + float(l[1]) + float(l[2])/60)

def get_index(header, string):
    logging.debug("header length: " +str(len(header)))
    for i in range(len(header)):
        logging.debug("header["+str(i)+"]"+header[i])
        if string in header[i] :
            return i

def get_std(array, save_header):
    num_training_set = 250000 # number of samples in training set
    percent_test_set = .1 # percent of training set used for testing
    num_test_set = num_training_set*percent_test_set
    batch_size_index = get_index(save_header,"batch_size")
    batch_size = array[batch_size_index]
    correction = array[-2]/(num_test_set/batch_size)**(.5)
    logging.debug("num test set: "+str(num_test_set))
    logging.debug("batch_size_index: "+str(batch_size_index))
    logging.debug("batch size: "+str(batch_size))
    logging.debug("correction: "+str(correction))
    return correction 
    # get corrected value


for file in files:
    file_count += 1
    print file
    with open(file,'r') as f:
        for line in f:
            word = rgx.findall(line)
            time = rgx2.findall(line)
            header = rgx3.findall(line)
            # Save word, time and header as going through file
            if word:
                save_word = word[0]
            if time:
                save_time = time[0]
            if header and (save_header==""):
               save_header = header[0] 
        # If all values have been extracted
            #logging.debug("save_word: "+str(save_word))
            #logging.debug("save_time: "+str(save_time))
            #logging.debug("save_header: "+str(save_header))
        if (save_header and save_time and save_word):
            good_file_count += 1 
            # On initialization save the header and start the array
            if (count == 0):
                # Save the header format (allows working with different models)
                save_header = save_header.split(",")
                logging.debug("save header length: ", len(save_header))
                logging.debug("saved header: "+str(save_header))
                # Create array
                array = np.fromstring(save_word,dtype = float, sep=',')
                # Add wall-time to array
                array[-1] = get_sec(save_time) 
                # Correction of standard deviation calculation (divides by sqrt(n))
                array[-2] = get_std(array, save_header)
                count = 1
                print(array)
            # After initialization 
            else: 
                word_array = np.fromstring(save_word,dtype = float, sep=',') 
                word_array[-1] = get_sec(save_time)
                word_array[-2] = get_std(word_array, save_header)
                array = np.vstack((array,word_array))
                print(word_array)
    # Refresh saved word and time
    save_word = None
    save_time = None
print("Number of files parsed: "+str(file_count))
print("Number of empty files: "+str(file_count-good_file_count))
cPickle.dump( array, open( "array_sda.p", "wb" ))
cPickle.dump( save_header, open("header_sda.p", "wb" ))
