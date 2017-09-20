#!/bin/sh

### Set the job name
#PBS -N time_it_5

### Set the project name, your department dc by default
#PBS -P physics

### Request email when job begins and ends
#PBS -m bea

### Specify email address to use for notification.
#PBS -M $USER@iitd.ac.in

####
#PBS -l select=1:ncpus=4
### Specify "wallclock time" required for this job, hhh:mm:ss

#PBS -l walltime=10:00:00

#PBS -o time_it_5_out.txt
#PBS -e time_it_5_err.txt

echo "==============================="
echo $PBS_JOBID
cat $PBS_NODEFILE
echo "==============================="
cd $PBS_O_WORKDIR

### sklearn_train
### state = 21
### --rand_forest     --include_0    -label(4,5,6,7)

echo ----------------------------
echo python -m himanshu.nn.sklearn_train -state=21 -label=4 --rand_forest 
python -m himanshu.nn.sklearn_train -state=21 -label=4 --rand_forest 
echo ----------------------------
echo python -m himanshu.nn.sklearn_train -state=21 -label=4 
python -m himanshu.nn.sklearn_train -state=21 -label=4 
echo ----------------------------
echo python -m himanshu.nn.sklearn_train -state=21 -label=5 --include_0 --rand_forest 
python -m himanshu.nn.sklearn_train -state=21 -label=5 --include_0 --rand_forest 
echo ----------------------------
echo python -m himanshu.nn.sklearn_train -state=21 -label=5 --include_0 
python -m himanshu.nn.sklearn_train -state=21 -label=5 --include_0 
echo ----------------------------
echo python -m himanshu.nn.sklearn_train -state=21 -label=5 --rand_forest 
python -m himanshu.nn.sklearn_train -state=21 -label=5 --rand_forest 
echo ----------------------------
echo python -m himanshu.nn.sklearn_train -state=21 -label=5 
python -m himanshu.nn.sklearn_train -state=21 -label=5 
echo ----------------------------
echo python -m himanshu.nn.sklearn_train -state=21 -label=6 --include_0 --rand_forest 
python -m himanshu.nn.sklearn_train -state=21 -label=6 --include_0 --rand_forest 
echo ----------------------------
echo python -m himanshu.nn.sklearn_train -state=21 -label=6 --include_0 
python -m himanshu.nn.sklearn_train -state=21 -label=6 --include_0 
echo ----------------------------
echo python -m himanshu.nn.sklearn_train -state=21 -label=6 --rand_forest 
python -m himanshu.nn.sklearn_train -state=21 -label=6 --rand_forest 
echo ----------------------------
echo python -m himanshu.nn.sklearn_train -state=21 -label=6 
python -m himanshu.nn.sklearn_train -state=21 -label=6 
echo ----------------------------
echo python -m himanshu.nn.sklearn_train -state=21 -label=7 --include_0 --rand_forest 
python -m himanshu.nn.sklearn_train -state=21 -label=7 --include_0 --rand_forest 
echo ----------------------------
echo python -m himanshu.nn.sklearn_train -state=21 -label=7 --include_0 
python -m himanshu.nn.sklearn_train -state=21 -label=7 --include_0 
echo ----------------------------
echo python -m himanshu.nn.sklearn_train -state=21 -label=7 --rand_forest 
python -m himanshu.nn.sklearn_train -state=21 -label=7 --rand_forest 
echo ----------------------------
echo python -m himanshu.nn.sklearn_train -state=21 -label=7 
python -m himanshu.nn.sklearn_train -state=21 -label=7 
echo ----------------------------
