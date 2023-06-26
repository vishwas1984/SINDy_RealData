#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=gp_mjo  #Set the job name to Example_MNMC_GPU
#SBATCH --time=03:00:00              #Set the wall clock limit to 1hr 30min
#SBATCH --nodes=2                    #Request 2 nodes
#SBATCH --ntasks-per-node=32         #Request 32 tasks/cores per node
#SBATCH --mem=2G                   #Request 248G (248GB) per node
#SBATCH --output=gp_mjo.%j #Redirect stdout/err to file
#SBATCH --partition=gpu              #Specify partition to submit job to
#SBATCH --gres=gpu:a100:1            #Specify GPU(s) per node, 1 A100 gpu 

##OPTIONAL JOB SPECIFICATIONS
##SBATCH --account=123456            #Set billing account to 123456
##SBATCH --mail-type=ALL             #Send email on all job events
##SBATCH --mail-user=email_address   #Send all emails to email_address

module purge
module load GCCcore/10.2.0 Python/3.8.6
module load GCC/10.2.0  CUDA/11.1.1  OpenMPI/4.0.5 GPyTorch/1.6.0
virtualenv --no-download gp_env
source gp_env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt