#Author: Jannicke Pearkes
#Purpose: Plot histograms of all hyperparameters

import subprocess

talk ="starting \n"
for t in range(2):
    for p in [1,2,3,5,6]: 
         talk += "python plot_histos2.py "+str(p)+" "+str(t)+";"  
print talk
subprocess.call(talk, shell = True)
