# Author: Jannicke Pearkes
# Purpose: Parses the output files from the batch scheduler for results


import re,sys,glob
import numpy as np
import cPickle
# What files to look for
#files = glob.glob('mlp_optimization.o2919*')
files = glob.glob('log_optimization.*')
rgx = re.compile(r"\{\[(.*?)\]\}") #TODO: Add in expression
rgx2 = re.compile(r"kb,walltime\=(.*)")
rgx3 = re.compile("\~\[(.*?)\]\~") #TODO: Add in expression
array = np.zeros((1,9))
count = 0
save_word = None
save_time = None
save_header = True

def getSec(s):
    l = s.split(':')
    return (float(l[0]) * 60 + float(l[1]) + float(l[2])/60)

for file in files:
    print file
    with open(file,'r') as f:
        for line in f:
            word = rgx.findall(line)
            time = rgx2.findall(line)
            header = rgx3.findall(line)
            #Added this because time and word are on different lines
            if word:
                save_word = word[0]
            if time:
                save_time = time[0]
            if header:
                save_header = header[0] 
        if (count == 0) and save_word and save_time and save_header:
                #save_header = save_header.split(",")
                array = np.fromstring(save_word,dtype = float, sep=',')
                array[-1] = getSec(save_time)
                count = 1
                #print save_header
                print array
        elif save_word and save_time: 
            word_array = np.fromstring(save_word,dtype = float, sep=',') 
            word_array[-1] = getSec(save_time)
            array = np.vstack((array,word_array))
            print word_array
    save_word = None
    save_time = None

cPickle.dump( array, open( "array.p", "wb" ))
cPickle.dump( save_header, open("header.p", "wb" ))
