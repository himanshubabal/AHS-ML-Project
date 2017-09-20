#!/bin/sh

### Set the job name
#PBS -N hot_cold_new

### Set the project name, your department dc by default
#PBS -P physics

### Request email when job begins and ends
#PBS -m bea

### Specify email address to use for notification.
#PBS -M $USER@iitd.ac.in

####
#PBS -l select=1:ncpus=1:ngpus=1
### Specify "wallclock time" required for this job, hhh:mm:ss

#PBS -l walltime=12:00:00

#PBS -o hot_cold_new_out.txt
#PBS -e hot_cold_new_error.txt

echo "==============================="
echo $PBS_JOBID
cat $PBS_NODEFILE
echo "==============================="
cd $PBS_O_WORKDIR

python nn_features_hotcold.py