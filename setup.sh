#!/bin/bash

# script to set up the simulations

# get focalplanes for SAT1
# requires s4sim (https://github.com/CMB-S4/s4sim) and toast
mkdir -p focalplanes
cd focalplanes
s4_hardware_to_toast3.py --telescope SAT1 --by-tube

# extract a CES from a sample schedule
schedule=schedules/atacama.1ces.txt
head -n 4 sample.txt >$schedule
