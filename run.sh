#!/bin/bash
#SBATCH --output=/cluster/home/tdieudonne/Lora-CL/logs/lora_cl_%j.out
#SBATCH --time=24:00:00
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpupr.24h
#SBATCH --gres=gpumem:35g
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16g

# ---------------------------
# Environment setup
# ---------------------------
USERNAME="${USERNAME:-tdieudonne}"
HOME_BASE="/cluster/home/${USERNAME}"
source "${HOME_BASE}/miniconda3/etc/profile.d/conda.sh"
conda init
module load eth_proxy
module load stack/2024-06 cuda/12.8.0
conda activate rds

#python3 main.py --config=./exps/sdlora_inr.json  >> InR.log 2>&1 &
#python3 main.py --config=./exps/sdlora_ina.json  >> InA.log 2>&1 &

python3 main.py --config=./exps/sdlora_c100.json
#python3 main.py --config=./exps/sdlora_inr.json