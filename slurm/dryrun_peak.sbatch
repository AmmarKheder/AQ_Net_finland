#!/bin/bash
#SBATCH --job-name=aqnet_peak
#SBATCH --account=project_462001140
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=64G
#SBATCH --time=02:30:00
#SBATCH --output=/scratch/project_462000640/ammar/aqnet_finland/dryrun_peak_%j.out

module load LUMI/25.03 partition/G rocm/6.0.3
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export MIOPEN_USER_DB_PATH=/tmp/miopen-$USER-$SLURM_JOB_ID
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
mkdir -p $MIOPEN_USER_DB_PATH

cd /scratch/project_462000640/ammar/aqnet_finland
source /scratch/project_462000640/ammar/aq_net2/venv_pytorch_rocm/bin/activate
# v2 Finland: loss anti-pics cran_pm (hotspot weight + penalite sous-estimation)
python3 -u main.py --hotspot_alpha 2.0 --under_penalty 2.0 \
        --epochs 100 --seq_lengths 48 --stride 6 --batch_size 128
echo "done $(date)"
