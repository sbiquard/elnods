#!/bin/bash
#SBATCH --job-name=elnod-array
#SBATCH --array=0-100%5   # max 5 jobs in the qos simultaneously
#SBATCH --account=nih@cpu
#SBATCH --qos=qos_cpu-dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --hint=nomultithread
#SBATCH --time=00:10:00
#SBATCH --output=out/logs/%x_%A_%a.out
#SBATCH --error=out/logs/%x_%A_%a.out

id=${SLURM_ARRAY_TASK_ID}

outdir=out
python3 gains.py --outdir $outdir --real $id --thinfp 4
