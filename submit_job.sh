#!/bin/bash
#SBATCH --job-name=wifi_xgboost_gpu   
#SBATCH --output=logs/job_%j.out      
#SBATCH --error=logs/job_%j.err       
#SBATCH --time=02:00:00               
#SBATCH --qos=acc_debug               # <-- CHANGED TO GPU PARTITION
#SBATCH --nodes=1                     
#SBATCH --ntasks=1                    
#SBATCH --cpus-per-task=16            
#SBATCH --gres=gpu:1                  # <-- CRITICAL: ASK FOR 1 GPU

# 1. Load the exact modules you used to build the .venv
module purge
module load intel
module load mkl
module load python/3.12.1

# 2. Activate your environment
source .venv/bin/activate

# 3. Ensure offline W&B mode inside the compute node
export WANDB_MODE=offline

# 4. Run your python script!
srun python main.py train config/xgboost_gpu.yaml