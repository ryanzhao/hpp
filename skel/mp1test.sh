#!/bin/bash
#----------------------------------------------------------------------------
# This is a very lousy script to run test on the provided dataset locally
# It assumes particular directory structure of dataset
#       e.g. ./dataset/mp1/[0-9]/*.raw
# "dataset" folder is parallel with "skel" and "build" folder
#============================================================
# Ryan (Weiran) Zhao 
# Started: Mon,Jan 13th 2014 10:12:17 PM EST
# Last Modified: Mon,Jan 13th 2014 10:14:24 PM EST
#----------------------------------------------------------------------------
BUILD=`dirname $PWD`/build
DATASET=`dirname $PWD`/dataset
MPNO=mp1
EXENO=MP1

for datasetNO in $(seq 0 9); do
    cp $DATASET/$MPNO/$datasetNO/* $BUILD
    $BUILD/$EXENO -i $BUILD/input0.raw,$BUILD/input1.raw -e $BUILD/output.raw -t vector
    # this is the "pause" function
    read -p "Test of dataset $datasetNO finished. Press [enter] key to continue"
done

# remove *.raw files from build
rm -f $BUILD/*.raw
