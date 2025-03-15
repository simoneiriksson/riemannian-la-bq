#!/bin/sh 
### General options 
### -- specify queue -- 
#BSUB -q gpuv100

### -- ask for number of cores (default: 1) -- 
#BSUB -n 1

### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"

### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"

### -- specify that we need 16GB of memory per core/slot -- 
#BSUB -R "rusage[mem=16GB]"

### -- set walltime limit: hh:mm -- 
#BSUB -W 23:00 

#BSUB -u simoneiriksson@gmail.com
## -- send notification at start -- 
#BSUB -B 
## -- send notification at completion -- 
#BSUB -N 

### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
### -- set the job Name -- 
#BSUB -J MyJob
#BSUB -o jobs/job%J.out
#BSUB -e jobs/job%J.err
echo start environment 
source /zhome/50/8/132685/envs/BNN_env/bin/activate
echo "Running script..."
python3 /zhome/50/8/132685/BNN/riemannian-la/riemannian_la/classification_test_experiment.py 


