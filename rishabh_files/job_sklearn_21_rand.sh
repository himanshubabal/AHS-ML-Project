#!/bin/sh

### Set the job name
#PBS -N 21_sk_rfor

### Set the project name, your department dc by default
#PBS -P physics

### Request email when job begins and ends
#PBS -m bea

### Specify email address to use for notification.
#PBS -M $USER@iitd.ac.in

####
#PBS -l select=1:ncpus=2
### Specify "wallclock time" required for this job, hhh:mm:ss

###PBS -l walltime=03:00:00
#PBS -l walltime=08:00:00

#PBS -o 21_sk_rfor_new_out.txt
#PBS -e 21_sk_rfor_new_err.txt

echo "==============================="
echo $PBS_JOBID
cat $PBS_NODEFILE
echo "==============================="
cd $PBS_O_WORKDIR

###echo "State-21  |  hotCold  |  include_0  |  random_forest"
###python -m himanshu.nn.sklearn_hotcold -state=21 --include_0 --rand_forest

###echo "____________________________________________________"
###echo "____________________________________________________"

echo "State-21  |  binary  |  include_0  |  random_forest"
###python -m himanshu.nn.sklearn_train -state=21 -label=1 --include_0 --rand_forest
###python -m himanshu.nn.sklearn_train -state=21 -label=2 --include_0 --rand_forest
###python -m himanshu.nn.sklearn_train -state=21 -label=3 --include_0 --rand_forest
###python -m himanshu.nn.sklearn_train -state=21 -label=4 --include_0 --rand_forest
###python -m himanshu.nn.sklearn_train -state=21 -label=5 --include_0 --rand_forest
echo "____________________________________________________"
echo "____________________________________________________"

python -m himanshu.nn.sklearn_train -state=21 -label=6 --include_0 --rand_forest
echo "____________________________________________________"
echo "____________________________________________________"

python -m himanshu.nn.sklearn_train -state=21 -label=7 --include_0 --rand_forest

echo "____________________________________________________"
echo "____________________________________________________"

echo "State-21  |  hotCold  |  random_forest"
python -m himanshu.nn.sklearn_hotcold -state=21 --rand_forest

echo "____________________________________________________"
echo "____________________________________________________"

echo "State-21  |  binary  |  random_forest  6 and 7"
python -m himanshu.nn.sklearn_train -state=21 -label=1 --rand_forest
echo "____________________________________________________"
echo "____________________________________________________"

python -m himanshu.nn.sklearn_train -state=21 -label=2 --rand_forest
echo "____________________________________________________"
echo "____________________________________________________"

python -m himanshu.nn.sklearn_train -state=21 -label=3 --rand_forest
echo "____________________________________________________"
echo "____________________________________________________"

python -m himanshu.nn.sklearn_train -state=21 -label=4 --rand_forest
echo "____________________________________________________"
echo "____________________________________________________"

python -m himanshu.nn.sklearn_train -state=21 -label=5 --rand_forest
echo "____________________________________________________"
echo "____________________________________________________"

python -m himanshu.nn.sklearn_train -state=21 -label=6 --rand_forest
echo "____________________________________________________"
echo "____________________________________________________"

python -m himanshu.nn.sklearn_train -state=21 -label=7 --rand_forest
echo "____________________________________________________"
echo "____________________________________________________"