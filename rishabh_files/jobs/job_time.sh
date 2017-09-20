#!/bin/sh

### Set the job name
#PBS -N time_it_1

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

#PBS -o time_it_1_out.txt
#PBS -e time_it_1_err.txt

echo "==============================="
echo $PBS_JOBID
cat $PBS_NODEFILE
echo "==============================="
cd $PBS_O_WORKDIR

### sklearn_hotcold
### state = 9, 21, 22
### --rand_forest   and  --include_0

echo python -m himanshu.nn.sklearn_hotcold -state=9 --include_0 --rand_forest 
python -m himanshu.nn.sklearn_hotcold -state=9 --include_0 --rand_forest 
echo ----------------------------
echo python -m himanshu.nn.sklearn_hotcold -state=9 --include_0 
python -m himanshu.nn.sklearn_hotcold -state=9 --include_0 
echo ----------------------------
echo python -m himanshu.nn.sklearn_hotcold -state=9 --rand_forest 
python -m himanshu.nn.sklearn_hotcold -state=9 --rand_forest 
echo ----------------------------
echo python -m himanshu.nn.sklearn_hotcold -state=9 
python -m himanshu.nn.sklearn_hotcold -state=9 
echo ----------------------------
echo python -m himanshu.nn.sklearn_hotcold -state=21 --include_0 --rand_forest 
python -m himanshu.nn.sklearn_hotcold -state=21 --include_0 --rand_forest 
echo ----------------------------
echo python -m himanshu.nn.sklearn_hotcold -state=21 --include_0 
python -m himanshu.nn.sklearn_hotcold -state=21 --include_0 
echo ----------------------------
echo python -m himanshu.nn.sklearn_hotcold -state=21 --rand_forest 
python -m himanshu.nn.sklearn_hotcold -state=21 --rand_forest 
echo ----------------------------
echo python -m himanshu.nn.sklearn_hotcold -state=21 
python -m himanshu.nn.sklearn_hotcold -state=21 
echo ----------------------------
echo python -m himanshu.nn.sklearn_hotcold -state=22 --include_0 --rand_forest 
python -m himanshu.nn.sklearn_hotcold -state=22 --include_0 --rand_forest 
echo ----------------------------
echo python -m himanshu.nn.sklearn_hotcold -state=22 --include_0 
python -m himanshu.nn.sklearn_hotcold -state=22 --include_0 
echo ----------------------------
echo python -m himanshu.nn.sklearn_hotcold -state=22 --rand_forest 
python -m himanshu.nn.sklearn_hotcold -state=22 --rand_forest 
echo ----------------------------
echo python -m himanshu.nn.sklearn_hotcold -state=22 
python -m himanshu.nn.sklearn_hotcold -state=22 
echo ----------------------------

echo python -m himanshu.nn.nn_features_hotcold -state=9 --include_0 
python -m himanshu.nn.nn_features_hotcold -state=9 --include_0 
echo ----------------------------
echo python -m himanshu.nn.nn_features_hotcold -state=9 
python -m himanshu.nn.nn_features_hotcold -state=9 
echo ----------------------------
echo python -m himanshu.nn.nn_features_hotcold -state=21 --include_0 
python -m himanshu.nn.nn_features_hotcold -state=21 --include_0 
echo ----------------------------
echo python -m himanshu.nn.nn_features_hotcold -state=21 
python -m himanshu.nn.nn_features_hotcold -state=21 
echo ----------------------------
echo python -m himanshu.nn.nn_features_hotcold -state=22 --include_0 
python -m himanshu.nn.nn_features_hotcold -state=22 --include_0 
echo ----------------------------
echo python -m himanshu.nn.nn_features_hotcold -state=22 
python -m himanshu.nn.nn_features_hotcold -state=22 
echo ----------------------------

