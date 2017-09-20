#!/bin/sh

### Set the job name
#PBS -N R

### Set the project name, your department dc by default
#PBS -P physics

### Request email when job begins and ends
#PBS -m bea

### Specify email address to use for notification.
#PBS -M $USER@iitd.ac.in

####
#PBS -l select=1:ncpus=4
### Specify "wallclock time" required for this job, hhh:mm:ss

#PBS -l walltime=24:00:00

#PBS -o r_out.txt
#PBS -e r_err.txt

echo "==============================="
echo $PBS_JOBID
cat $PBS_NODEFILE
echo "==============================="
cd $PBS_O_WORKDIR

Rscript ahs.r

echo ""
echo ""
echo "==============================="
echo "==============================="
echo "==============================="
echo "==============================="
echo "==============================="
echo ""
echo ""

Rscript ahs_new.r
