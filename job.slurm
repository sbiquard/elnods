#!/bin/bash
#SBATCH --job-name=elnod-test
#SBATCH --account=nih@cpu
#SBATCH --qos=qos_cpu-dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --hint=nomultithread
#SBATCH --time=00:10:00
#SBATCH --output=out/logs/%x_%j.out
#SBATCH --error=out/logs/%x_%j.out

outdir=out/test
python3 gains.py --outdir $outdir --thinfp 4
