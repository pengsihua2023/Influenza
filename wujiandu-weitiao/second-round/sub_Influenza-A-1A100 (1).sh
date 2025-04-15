#!/bin/bash
#SBATCH --job-name=Influenza-1A100           # Job name
#SBATCH --partition=gpu_p             # Partition (queue) name
#SBATCH --gres=gpu:A100:1             # Requests 4 GPU devices
#SBATCH --nodes=1                     # Requests 1 node
#SBATCH --ntasks=1                    # Total number of tasks
#SBATCH --ntasks-per-node=1           # Number of tasks per node
#SBATCH --cpus-per-task=4            # Number of CPU cores per task
#SBATCH --mem=256gb                   # Job memory request
#SBATCH --time=7-00:00:00             # Time limit hrs:min:sec
#SBATCH --output=Influenza.%j.out          # Standard output log
#SBATCH --error=Influenza.%j.err           # Standard error log
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=sp96859@uga.edu   # Where to send mail

cd $SLURM_SUBMIT_DIR

ml Miniconda3/23.5.2-0
#ml PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
# ml PyTorch/2.1.2-foss-2022a-CUDA-12.1.1
ml CUDA/12.2.0
module load gcc/9.3.0

echo "Checking allocated GPUs:"
nvidia-smi

echo "Activating Conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /home/sp96859/.conda/envs/ESM2

# Use DeepSpeed launcher to start the script
deepspeed --num_gpus=1 Influenza-old-code-modifyed.py
