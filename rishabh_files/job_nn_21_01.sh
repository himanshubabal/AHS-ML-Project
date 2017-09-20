#!/bin/sh

### Set the job name
#PBS -N 21_nn_1

### Set the project name, your department dc by default
#PBS -P physics

### Request email when job begins and ends
#PBS -m bea

### Specify email address to use for notification.
#PBS -M $USER@iitd.ac.in

####
#PBS -l select=1:ncpus=2:ngpus=1
### Specify "wallclock time" required for this job, hhh:mm:ss

#PBS -l walltime=14:00:00

#PBS -o 21_nn_01_out.txt
#PBS -e 21_nn_01_err.txt

echo "==============================="
echo $PBS_JOBID
cat $PBS_NODEFILE
echo "==============================="
cd $PBS_O_WORKDIR

echo "State-21  |  binary - 1 |  include_0  |  nn"
python -m himanshu.nn.nn_feature_binary_arg -state=21 -label=1 --include_0

echo "___________________________________________"
echo "___________________________________________"

echo "State-21  |  binary - 1  |  nn"
python -m himanshu.nn.nn_feature_binary_arg -state=21 -label=1
