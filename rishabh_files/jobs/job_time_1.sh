#!/bin/sh

### Set the job name
#PBS -N time_it_2

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

#PBS -o time_it_2_out.txt
#PBS -e time_it_2_err.txt

echo "==============================="
echo $PBS_JOBID
cat $PBS_NODEFILE
echo "==============================="
cd $PBS_O_WORKDIR

### sklearn_train
### state = 9
### --rand_forest     --include_0    -label(1,2,3,4)

echo python -m himanshu.nn.sklearn_train -state=9 -label=1 --include_0 --rand_forest 
python -m himanshu.nn.sklearn_train -state=9 -label=1 --include_0 --rand_forest 
echo ----------------------------
echo python -m himanshu.nn.sklearn_train -state=9 -label=1 --include_0 
python -m himanshu.nn.sklearn_train -state=9 -label=1 --include_0 
echo ----------------------------
echo python -m himanshu.nn.sklearn_train -state=9 -label=1 --rand_forest 
python -m himanshu.nn.sklearn_train -state=9 -label=1 --rand_forest 
echo ----------------------------
echo python -m himanshu.nn.sklearn_train -state=9 -label=1 
python -m himanshu.nn.sklearn_train -state=9 -label=1 
echo ----------------------------
echo python -m himanshu.nn.sklearn_train -state=9 -label=2 --include_0 --rand_forest 
python -m himanshu.nn.sklearn_train -state=9 -label=2 --include_0 --rand_forest 
echo ----------------------------
echo python -m himanshu.nn.sklearn_train -state=9 -label=2 --include_0 
python -m himanshu.nn.sklearn_train -state=9 -label=2 --include_0 
echo ----------------------------
echo python -m himanshu.nn.sklearn_train -state=9 -label=2 --rand_forest 
python -m himanshu.nn.sklearn_train -state=9 -label=2 --rand_forest 
echo ----------------------------
echo python -m himanshu.nn.sklearn_train -state=9 -label=2 
python -m himanshu.nn.sklearn_train -state=9 -label=2 
echo ----------------------------
echo python -m himanshu.nn.sklearn_train -state=9 -label=3 --include_0 --rand_forest 
python -m himanshu.nn.sklearn_train -state=9 -label=3 --include_0 --rand_forest 
echo ----------------------------
echo python -m himanshu.nn.sklearn_train -state=9 -label=3 --include_0 
python -m himanshu.nn.sklearn_train -state=9 -label=3 --include_0 
echo ----------------------------
echo python -m himanshu.nn.sklearn_train -state=9 -label=3 --rand_forest 
python -m himanshu.nn.sklearn_train -state=9 -label=3 --rand_forest 
echo ----------------------------
echo python -m himanshu.nn.sklearn_train -state=9 -label=3 
python -m himanshu.nn.sklearn_train -state=9 -label=3 
echo ----------------------------
echo python -m himanshu.nn.sklearn_train -state=9 -label=4 --include_0 --rand_forest 
python -m himanshu.nn.sklearn_train -state=9 -label=4 --include_0 --rand_forest 
echo ----------------------------
echo python -m himanshu.nn.sklearn_train -state=9 -label=4 --include_0 
python -m himanshu.nn.sklearn_train -state=9 -label=4 --include_0 
