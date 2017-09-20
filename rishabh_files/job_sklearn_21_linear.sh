#!/bin/sh

### Set the job name
#PBS -N 21_sk_linear

### Set the project name, your department dc by default
#PBS -P physics

### Request email when job begins and ends
#PBS -m bea

### Specify email address to use for notification.
#PBS -M $USER@iitd.ac.in

####
#PBS -l select=1:ncpus=2
### Specify "wallclock time" required for this job, hhh:mm:ss

###PBS -l walltime=16:00:00
#PBS -l walltime=20:00:00

#PBS -o 21_sk_linear_new_out.txt
#PBS -e 21_sk_linear_new_err.txt

echo "==============================="
echo $PBS_JOBID
cat $PBS_NODEFILE
echo "==============================="
cd $PBS_O_WORKDIR

###echo "State-21  |  hotCold  |  include_0  |  linear_regression"
###python -m himanshu.nn.sklearn_hotcold -state=21 --include_0

###echo "____________________________________________________"
###echo "____________________________________________________"

###echo "State-21  |  binary  |  include_0  |  linear_regression"
###python -m himanshu.nn.sklearn_train -state=21 -label=1 --include_0
###python -m himanshu.nn.sklearn_train -state=21 -label=2 --include_0
###python -m himanshu.nn.sklearn_train -state=21 -label=3 --include_0
###python -m himanshu.nn.sklearn_train -state=21 -label=4 --include_0
###python -m himanshu.nn.sklearn_train -state=21 -label=5 --include_0
###python -m himanshu.nn.sklearn_train -state=21 -label=6 --include_0
###python -m himanshu.nn.sklearn_train -state=21 -label=7 --include_0

###echo "____________________________________________________"
###echo "____________________________________________________"

echo "State-21  |  hotCold  |  linear_regression"
python -m himanshu.nn.sklearn_hotcold -state=21

echo "____________________________________________________"
echo "____________________________________________________"

echo "State-21  |  binary  |  linear_regression"
python -m himanshu.nn.sklearn_train -state=21 -label=1
echo "____________________________________________________"
echo "____________________________________________________"

python -m himanshu.nn.sklearn_train -state=21 -label=2
echo "____________________________________________________"
echo "____________________________________________________"

python -m himanshu.nn.sklearn_train -state=21 -label=3
echo "____________________________________________________"
echo "____________________________________________________"

python -m himanshu.nn.sklearn_train -state=21 -label=4
echo "____________________________________________________"
echo "____________________________________________________"

python -m himanshu.nn.sklearn_train -state=21 -label=5
echo "____________________________________________________"
echo "____________________________________________________"

python -m himanshu.nn.sklearn_train -state=21 -label=6
echo "____________________________________________________"
echo "____________________________________________________"

python -m himanshu.nn.sklearn_train -state=21 -label=7
echo "____________________________________________________"
echo "____________________________________________________"

