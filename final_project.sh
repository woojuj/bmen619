#!/bin/zsh

#SBATCH --job-name=619
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64GB
#SBATCH --time=10:00:00
#SBATCH --output=/home/wooju.chung/bmen619/output/Output_%j.out

##SBATCH --gres=gpu:1
##SBATCH --partition=bigmem
##SBATCH --partition=gpu-a100

#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=wooju.chung@ucalgary.ca 

eval "$(/home/wooju.chung/software/miniforge3/bin/conda shell.bash hook)"
conda activate bmen619

# Check the GPUs with the nvidia-smi command.
# nvidia-smi 

JOB_ID=$SLURM_JOB_ID  # job id for environment variable

python /home/wooju.chung/bmen619/main_code.py "$JOB_ID"