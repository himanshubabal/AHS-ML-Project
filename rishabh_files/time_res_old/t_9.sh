#!/bin/sh

### Set the job name
#PBS -N t_9

### Set the project name, your department dc by default
#PBS -P physics

### Request email when job begins and ends
#PBS -m bea

### Specify email address to use for notification.
#PBS -M $USER@iitd.ac.in

####
#PBS -l select=1:ncpus=2:ngpus=1
### Specify "wallclock time" required for this job, hhh:mm:ss

#PBS -l walltime=24:00:00

#PBS -o t_9_out.txt
#PBS -e t_9_err.txt

echo "==============================="
echo $PBS_JOBI
cat $PBS_NODEFILE
echo "==============================="
cd $PBS_O_WORKDIR


### 9

### 10 min/iteration -- nn
### 30 min/cmd -- nn

### 0.5 min/iteration -- rand_for
### 1 min/cmd -- nn

### t = 30 + 15 + 30x7 + 5 + 30x7
###   = 8 hours

echo "python -m himanshu.nn.nn_features_hotcold -state=9"
python -m himanshu.nn.nn_features_hotcold -state=9 
echo "---------------------------"

echo "____________________________________________________________________________"
echo "____________________________________________________________________________"


echo "python -m himanshu.nn.sklearn_train -state=9 -label=1 --rand_forest"
python -m himanshu.nn.sklearn_train -state=9 -label=1 --rand_forest 
echo "---------------------------"
echo "python -m himanshu.nn.sklearn_train -state=9 -label=2 --rand_forest"
python -m himanshu.nn.sklearn_train -state=9 -label=2 --rand_forest 
echo "---------------------------"
echo "python -m himanshu.nn.sklearn_train -state=9 -label=3 --rand_forest"
python -m himanshu.nn.sklearn_train -state=9 -label=3 --rand_forest 
echo "---------------------------"
echo "python -m himanshu.nn.sklearn_train -state=9 -label=4 --rand_forest"
python -m himanshu.nn.sklearn_train -state=9 -label=4 --rand_forest 
echo "---------------------------"
echo "python -m himanshu.nn.sklearn_train -state=9 -label=5 --rand_forest"
python -m himanshu.nn.sklearn_train -state=9 -label=5 --rand_forest 
echo "---------------------------"
echo "python -m himanshu.nn.sklearn_train -state=9 -label=6 --rand_forest"
python -m himanshu.nn.sklearn_train -state=9 -label=6 --rand_forest 
echo "---------------------------"
echo "python -m himanshu.nn.sklearn_train -state=9 -label=7 --rand_forest"
python -m himanshu.nn.sklearn_train -state=9 -label=7 --rand_forest 
echo "---------------------------"

echo "____________________________________________________________________________"
echo "____________________________________________________________________________"



echo "python -m himanshu.nn.sklearn_train -state=9 -label=1"
python -m himanshu.nn.sklearn_train -state=9 -label=1 
echo "---------------------------"
echo "python -m himanshu.nn.sklearn_train -state=9 -label=2"
python -m himanshu.nn.sklearn_train -state=9 -label=2 
echo "---------------------------"
echo "python -m himanshu.nn.sklearn_train -state=9 -label=3"
python -m himanshu.nn.sklearn_train -state=9 -label=3 
echo "---------------------------"
echo "python -m himanshu.nn.sklearn_train -state=9 -label=4"
python -m himanshu.nn.sklearn_train -state=9 -label=4 
echo "---------------------------"
echo "python -m himanshu.nn.sklearn_train -state=9 -label=5"
python -m himanshu.nn.sklearn_train -state=9 -label=5 
echo "---------------------------"
echo "python -m himanshu.nn.sklearn_train -state=9 -label=6"
python -m himanshu.nn.sklearn_train -state=9 -label=6 
echo "---------------------------"
echo "python -m himanshu.nn.sklearn_train -state=9 -label=7"
python -m himanshu.nn.sklearn_train -state=9 -label=7 
echo "---------------------------"

echo "____________________________________________________________________________"
echo "____________________________________________________________________________"


echo "python -m himanshu.nn.sklearn_hotcold -state=9 --rand_forest"
python -m himanshu.nn.sklearn_hotcold -state=9 --rand_forest 
echo "---------------------------"
echo "python -m himanshu.nn.sklearn_hotcold -state=9"
python -m himanshu.nn.sklearn_hotcold -state=9 
echo "---------------------------"

echo "____________________________________________________________________________"
echo "____________________________________________________________________________"


echo "python -m himanshu.nn.nn_feature_binary_arg -state=9 -label=1"
python -m himanshu.nn.nn_feature_binary_arg -state=9 -label=1 
echo "---------------------------"
echo "python -m himanshu.nn.nn_feature_binary_arg -state=9 -label=2"
python -m himanshu.nn.nn_feature_binary_arg -state=9 -label=2 
echo "---------------------------"
echo "python -m himanshu.nn.nn_feature_binary_arg -state=9 -label=3"
python -m himanshu.nn.nn_feature_binary_arg -state=9 -label=3 
echo "---------------------------"
echo "python -m himanshu.nn.nn_feature_binary_arg -state=9 -label=4"
python -m himanshu.nn.nn_feature_binary_arg -state=9 -label=4 
echo "---------------------------"
echo "python -m himanshu.nn.nn_feature_binary_arg -state=9 -label=5"
python -m himanshu.nn.nn_feature_binary_arg -state=9 -label=5 
echo "---------------------------"
echo "python -m himanshu.nn.nn_feature_binary_arg -state=9 -label=6"
python -m himanshu.nn.nn_feature_binary_arg -state=9 -label=6 
echo "---------------------------"
echo "python -m himanshu.nn.nn_feature_binary_arg -state=9 -label=7"
python -m himanshu.nn.nn_feature_binary_arg -state=9 -label=7 
echo "--------------------------"


echo "____________________________________________________________________________"
echo "____________________________________________________________________________"

