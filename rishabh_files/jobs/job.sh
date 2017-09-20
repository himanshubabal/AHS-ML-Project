#!/bin/sh

### Set the job name
#PBS -N prep_data

### Set the project name, your department dc by default
#PBS -P physics

### Request email when job begins and ends
#PBS -m bea

### Specify email address to use for notification.
#PBS -M $USER@iitd.ac.in

####
#PBS -l select=1:ncpus=12
### Specify "wallclock time" required for this job, hhh:mm:ss

#PBS -l walltime=04:00:00

#PBS -o prep_data_out.txt
#PBS -e prep_data_error.txt

echo "==============================="
echo $PBS_JOBID
cat $PBS_NODEFILE
echo "==============================="
cd $PBS_O_WORKDIR

python -m himanshu.process.process_all -state=22 -force_all=False -include_0=True
python -m himanshu.process.process_all -state=22 -force_all=False -include_0=False

python -m himanshu.process.process_all -state=21 -force_all=True -include_0=True
python -m himanshu.process.process_all -state=21 -force_all=True -include_0=False

python -m himanshu.process.process_all -state=9 -force_all=True -include_0=True
python -m himanshu.process.process_all -state=9 -force_all=True -include_0=False

python -m himanshu.process.process_all -state=5 -force_all=True -include_0=True
python -m himanshu.process.process_all -state=5 -force_all=True -include_0=False

python -m himanshu.process.process_all -state=8 -force_all=True -include_0=True
python -m himanshu.process.process_all -state=8 -force_all=True -include_0=False

python -m himanshu.process.process_all -state=10 -force_all=True -include_0=True
python -m himanshu.process.process_all -state=10 -force_all=True -include_0=False

python -m himanshu.process.process_all -state=18 -force_all=True -include_0=True
python -m himanshu.process.process_all -state=18 -force_all=True -include_0=False

python -m himanshu.process.process_all -state=20 -force_all=True -include_0=True
python -m himanshu.process.process_all -state=20 -force_all=True -include_0=False

python -m himanshu.process.process_all -state=23 -force_all=True -include_0=True
python -m himanshu.process.process_all -state=23 -force_all=True -include_0=False
