#!/bin/bash
#SBATCH --job-name=elnod
#SBATCH --account=nih@cpu
#SBATCH --qos=qos_cpu-dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --hint=nomultithread
#SBATCH --time=00:10:00
#SBATCH --output=out/elnod%j.out
#SBATCH --error=out/elnod%j.out

cd ${SLURM_SUBMIT_DIR}

module purge
cmbenv

start=$1
stop=$2

outdir=out

echo "Running from $start to $stop"

for ((real = $start; real < $stop; real++)); do
    echo ""
    echo "----------------------------------------"
    echo "real = $real"
    python3 gains.py --outdir $outdir --real $real --thinfp 4
done

echo "Done!"
