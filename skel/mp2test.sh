#!/bin/bash
#----------------------------------------------------------------------------
# Adapted from ./hpp/skel/mp1test.sh directly
#============================================================
# Ryan (Weiran) Zhao 
# Started: Mon,Jan 13th 2014 10:12:17 PM EST
# Modified: Tue,Jan 14th 2014 10:09:21 PM EST
#           Adapted from mp1test.sh to fit testing for mp2
# Last Modified: Tue,Jan 14th 2014 10:10:46 PM EST
#----------------------------------------------------------------------------
BUILD=`dirname $PWD`/build
DATASET=`dirname $PWD`/dataset
MPNO=mp2
EXENO=MP2

for datasetNO in $(seq 0 9); do
    cp $DATASET/$MPNO/$datasetNO/* $BUILD
    $BUILD/$EXENO -i $BUILD/input0.raw,$BUILD/input1.raw -e $BUILD/output.raw -t matrix
    # this is the "pause" function
    read -p "Test of dataset $datasetNO finished. Press [enter] key to continue"
done

# remove *.raw files from build
rm -f $BUILD/*.raw
