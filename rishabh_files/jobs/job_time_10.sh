#!/bin/sh

### Set the job name
#PBS -N time_it_10

### Set the project name, your department dc by default
#PBS -P physics

### Request email when job begins and ends
#PBS -m bea

### Specify email address to use for notification.
#PBS -M $USER@iitd.ac.in

####
#PBS -l select=1:ncpus=2:ngpus=1
### Specify "wallclock time" required for this job, hhh:mm:ss

#PBS -l walltime=15:00:00

#PBS -o time_it_10_out.txt
#PBS -e time_it_10_err.txt

echo "==============================="
echo $PBS_JOBID
cat $PBS_NODEFILE
echo "==============================="
cd $PBS_O_WORKDIR

### nn_feature_binary_arg
### state = 21
### -label

echo python -m himanshu.nn.nn_feature_binary_arg -state=21 -label=1 
python -m himanshu.nn.nn_feature_binary_arg -state=21 -label=1 
echo ----------------------------
echo python -m himanshu.nn.nn_feature_binary_arg -state=21 -label=2 
python -m himanshu.nn.nn_feature_binary_arg -state=21 -label=2 
echo ----------------------------
echo python -m himanshu.nn.nn_feature_binary_arg -state=21 -label=3 
python -m himanshu.nn.nn_feature_binary_arg -state=21 -label=3 
echo ----------------------------
echo python -m himanshu.nn.nn_feature_binary_arg -state=21 -label=4 
python -m himanshu.nn.nn_feature_binary_arg -state=21 -label=4 
echo ----------------------------
echo python -m himanshu.nn.nn_feature_binary_arg -state=21 -label=5 
python -m himanshu.nn.nn_feature_binary_arg -state=21 -label=5 
echo ----------------------------
echo python -m himanshu.nn.nn_feature_binary_arg -state=21 -label=6 
python -m himanshu.nn.nn_feature_binary_arg -state=21 -label=6 
echo ----------------------------
echo python -m himanshu.nn.nn_feature_binary_arg -state=21 -label=7 
python -m himanshu.nn.nn_feature_binary_arg -state=21 -label=7 
echo ----------------------------