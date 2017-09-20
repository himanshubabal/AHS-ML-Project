#!/bin/sh

### Set the job name
#PBS -N 9_0_hc

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

#PBS -o 9_0_hc_out.txt
#PBS -e 9_0_hc_err.txt

echo "==============================="
echo $PBS_JOBID
cat $PBS_NODEFILE
echo "==============================="
cd $PBS_O_WORKDIR

python -m himanshu.nn.nn_features_hotcold -state=9 -include_0=True
python -m himanshu.nn.nn_features_hotcold -state=9 -include_0=False

