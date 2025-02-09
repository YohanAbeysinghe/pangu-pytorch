#!/bin/bash
#SBATCH --job-name=pa_do_ar
#SBATCH --account=project_462000472
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --partition=small-g
#SBATCH --output=/pfs/lustrep1/scratch/project_462000472/akhtar/climate_modeling/pangu-pm25/scripts/outputs/pangu_download_%A_%a.log
#SBATCH --array=0-5

# Initialize Conda and activate the environment
source /users/akhtarmunir/miniconda3/etc/profile.d/conda.sh
conda activate /pfs/lustrep1/scratch/project_462000472/akhtar/envs/pangu

# Export the CDSAPI_RC environment variable
export CDSAPI_RC=/pfs/lustrep1/scratch/project_462000472/akhtar/climate_modeling/.cdsapirc

# Map the SLURM_ARRAY_TASK_ID to the correct year
year=$((2003 + SLURM_ARRAY_TASK_ID))

# Run your Python script, passing the year based on the SLURM_ARRAY_TASK_ID
srun python /pfs/lustrep1/scratch/project_462000472/akhtar/climate_modeling/pangu-pytorch/data_download/legacy/upper.py --year ${year}