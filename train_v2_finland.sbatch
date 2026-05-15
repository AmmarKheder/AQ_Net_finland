#!/bin/bash
#SBATCH --job-name=aqnet_v2
#SBATCH --account=project_462001140
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8           # 1 rang DDP par GCD
#SBATCH --cpus-per-task=7
#SBATCH --mem=480G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/project_462000640/ammar/aqnet_finland/aqnet_v2_%j.out

# v2 Finland: 1 noeud x 8 GPU (MI250X) LUMI, DDP via torchrun
module load LUMI/25.03 partition/G rocm/6.0.3
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export MIOPEN_USER_DB_PATH=/tmp/miopen-$USER-$SLURM_JOB_ID
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
srun --ntasks-per-node=1 mkdir -p $MIOPEN_USER_DB_PATH

cd /scratch/project_462000640/ammar/aqnet_finland
source /scratch/project_462000640/ammar/aq_net2/venv_pytorch_rocm/bin/activate

# NB: necessite la version DDP de main.py (etape suivante).
# Mono-GPU de validation : srun -n1 --gpus 1 python3 -u main.py --epochs 5 --seq_lengths 24
srun python3 -u main.py --epochs "${EPOCHS:-50}" --batch_size "${BS:-64}" --stride "${STRIDE:-6}"
echo "done $(date)"
