#!/bin/sh

### Set the job name
#PBS -N t_22_i

### Set the project name, your department dc by default
#PBS -P physics

### Request email when job begins and ends
#PBS -m bea

### Specify email address to use for notification.
#PBS -M $USER@iitd.ac.in

####
#PBS -l select=1:ncpus=24:ngpus=1
### Specify "wallclock time" required for this job, hhh:mm:ss

#PBS -l walltime=12:00:00

#PBS -o t_22_i_new_out.txt
#PBS -e t_22_i_new_err.txt

echo "==============================="
echo $PBS_JOBI
cat $PBS_NODEFILE
echo "==============================="
cd $PBS_O_WORKDIR


### 22 inc

### 10 min/iteration -- nn
### 30 min/cmd -- nn

### 0.5 min/iteration -- rand_for
### 1 min/cmd -- nn

### t = 30 + 15 + 30x7 + 5 + 30x7
###   = 8 hours

echo "____________________________________________________________________________"
echo "____________________________________________________________________________"


echo "python -m himanshu.nn.nn_feature_binary_arg -state=22 -label=1 --include_0"
python -m himanshu.nn.nn_feature_binary_arg -state=22 -label=1 --include_0 
echo "---------------------------"
echo "____________________________________________________________________________"
echo "____________________________________________________________________________"

echo "python -m himanshu.nn.nn_feature_binary_arg -state=22 -label=2 --include_0"
python -m himanshu.nn.nn_feature_binary_arg -state=22 -label=2 --include_0 
echo "---------------------------"
echo "____________________________________________________________________________"
echo "____________________________________________________________________________"

echo "python -m himanshu.nn.nn_feature_binary_arg -state=22 -label=3 --include_0"
python -m himanshu.nn.nn_feature_binary_arg -state=22 -label=3 --include_0 
echo "---------------------------"
echo "____________________________________________________________________________"
echo "____________________________________________________________________________"

echo "python -m himanshu.nn.nn_feature_binary_arg -state=22 -label=4 --include_0"
python -m himanshu.nn.nn_feature_binary_arg -state=22 -label=4 --include_0 
echo "---------------------------"
echo "____________________________________________________________________________"
echo "____________________________________________________________________________"

echo "python -m himanshu.nn.nn_feature_binary_arg -state=22 -label=5 --include_0"
python -m himanshu.nn.nn_feature_binary_arg -state=22 -label=5 --include_0 
echo "---------------------------"
echo "____________________________________________________________________________"
echo "____________________________________________________________________________"

echo "python -m himanshu.nn.nn_feature_binary_arg -state=22 -label=6 --include_0"
python -m himanshu.nn.nn_feature_binary_arg -state=22 -label=6 --include_0 
echo "---------------------------"
echo "____________________________________________________________________________"
echo "____________________________________________________________________________"

echo "python -m himanshu.nn.nn_feature_binary_arg -state=22 -label=7 --include_0"
python -m himanshu.nn.nn_feature_binary_arg -state=22 -label=7 --include_0 
echo "--------------------"
echo "____________________________________________________________________________"
echo "____________________________________________________________________________"
