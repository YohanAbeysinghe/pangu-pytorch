#!/bin/bash
#SBATCH --job-name=pa_tr
#SBATCH --account=project_462000472
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --partition=small-g
#SBATCH --output=/pfs/lustrep1/scratch/project_462000472/akhtar/climate_modeling/pangu-pm25-ddp/scripts/outputs/Jan_22_2/pangu_train.o%j
#SBATCH --error=/pfs/lustrep1/scratch/project_462000472/akhtar/climate_modeling/pangu-pm25-ddp/scripts/outputs/Jan_22_2/pangu_train.e%j

# Initialize Conda and activate the environment
source /users/akhtarmunir/miniconda3/etc/profile.d/conda.sh
conda activate /pfs/lustrep1/scratch/project_462000472/akhtar/envs/pangu

# Activate Wandb
export NETRC=/pfs/lustrep1/scratch/project_462000472/akhtar/climate_modeling/.netrc

# Run your Python script
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 /pfs/lustrep1/scratch/project_462000472/akhtar/climate_modeling/pangu-pm25-ddp/finetune/finetune_pm25_multi.py