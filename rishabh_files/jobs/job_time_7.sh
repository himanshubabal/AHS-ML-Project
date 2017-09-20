#!/bin/sh

### Set the job name
#PBS -N time_it_7

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

#PBS -o time_it_7_out.txt
#PBS -e time_it_7_err.txt

echo "==============================="
echo $PBS_JOBID
cat $PBS_NODEFILE
echo "==============================="
cd $PBS_O_WORKDIR

### nn_feature_binary_arg
### state = 9
### -label  and  --include_0

echo python -m himanshu.nn.nn_feature_binary_arg -state=9 -label=1 --include_0 
python -m himanshu.nn.nn_feature_binary_arg -state=9 -label=1 --include_0 
echo ----------------------------
echo python -m himanshu.nn.nn_feature_binary_arg -state=9 -label=2 --include_0 
python -m himanshu.nn.nn_feature_binary_arg -state=9 -label=2 --include_0 
echo ----------------------------
echo python -m himanshu.nn.nn_feature_binary_arg -state=9 -label=3 --include_0 
python -m himanshu.nn.nn_feature_binary_arg -state=9 -label=3 --include_0 
echo ----------------------------
echo python -m himanshu.nn.nn_feature_binary_arg -state=9 -label=4 --include_0 
python -m himanshu.nn.nn_feature_binary_arg -state=9 -label=4 --include_0 
echo ----------------------------
echo python -m himanshu.nn.nn_feature_binary_arg -state=9 -label=5 --include_0 
python -m himanshu.nn.nn_feature_binary_arg -state=9 -label=5 --include_0 
echo ----------------------------
echo python -m himanshu.nn.nn_feature_binary_arg -state=9 -label=6 --include_0 
python -m himanshu.nn.nn_feature_binary_arg -state=9 -label=6 --include_0 
echo ----------------------------
echo python -m himanshu.nn.nn_feature_binary_arg -state=9 -label=7 --include_0 
python -m himanshu.nn.nn_feature_binary_arg -state=9 -label=7 --include_0 
echo ----------------------------

