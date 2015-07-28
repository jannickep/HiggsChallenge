#!/bin/sh
# Script for running serial program, sda_batch.py

#PBS -N linear_search
#PBS -l walltime=0:20:00
#PBS -l mem=4gb
#PBS -l procs=1
#PBS -j oe
#PBS -r n

cd /home/jpearkes/HiggsChallenge/code
echo "Current working directory is `pwd`"
module load python 
echo "Starting run at: `date`"
python compare_classifiers.py 5
echo "Program mlp finished with exit code $? at: `date`"
