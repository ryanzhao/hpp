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
# Modified: Tue,Jan 28th 2014 02:26:23 PM EST
#           Adpated from mp3test.sh for mp4
# Last Modified: Tue,Jan 28th 2014 02:57:20 PM EST
#----------------------------------------------------------------------------
BUILD=`dirname $PWD`/build
DATASET=`dirname $PWD`/dataset
MPNO=mp4
EXENO=MP4

for datasetNO in $(seq 0 9); do
    cp $DATASET/$MPNO/$datasetNO/* $BUILD
    $BUILD/$EXENO -i $BUILD/input0.raw -e $BUILD/output.raw -t vector
    # this is the "pause" function
    read -p "Test of dataset $datasetNO finished. Press [enter] key to continue"
done

# remove *.raw files from build
rm -f $BUILD/*.raw
