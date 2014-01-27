#!/bin/bash
#----------------------------------------------------------------------------
# Adapted from ./hpp/skel/mp2test.sh directly
#============================================================
# Ryan (Weiran) Zhao 
# Started: Mon,Jan 13th 2014 10:12:17 PM EST
# Modified: Tue,Jan 14th 2014 10:09:21 PM EST
#           Adapted from mp1test.sh to fit testing for mp2
# Modified: Sat,Jan 18th 2014 10:40:36 PM EST
#           Adapted from mp2test.sh to fit testing for mp3
# Modified: Sat,Jan 25th 2014 03:33:05 PM EST
#           Adapted from mp3test.sh to fit testing for mp6
# Last Modified: Sat,Jan 25th 2014 04:17:14 PM EST
#----------------------------------------------------------------------------
BUILD=`dirname $PWD`/build
DATASET=`dirname $PWD`/dataset
MPNO=mp6
EXENO=MP6

for datasetNO in $(seq 0 9); do
    cp $DATASET/$MPNO/$datasetNO/* $BUILD
    $BUILD/$EXENO -i $BUILD/input0.ppm,$BUILD/input1.csv -e $BUILD/output.ppm -t image
    # this is the "pause" function
    read -p "Test of dataset $datasetNO finished. Press [enter] key to continue"
done

# remove *.raw files from build
rm -f $BUILD/*.csv
rm -f $BUILD/*.ppm
