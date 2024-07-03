#!/bin/bash

start=$1
stop=$2

outdir=elnod_out

logfile=${outdir}/mc_${start}_${stop}.log
if [[ -e $logfile ]]; then
    echo "$logfile exists. Overwriting!"
else
    echo "Writing to $logfile"
    touch $logfile
fi

echo "Running MC from $start to $stop" >$logfile

for ((real = $start; real < $stop; real++)); do
    echo "" >>$logfile
    echo "----------------------------------------" >>$logfile
    echo "real = $real" >>$logfile
    python3 gains.py --outdir $outdir --real $real >>$logfile 2>&1
done

echo "Done! Check $logfile for details."
