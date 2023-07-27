#!/bin/bash

module purge
module load horton/2.1.1
horton-atomdb.py input g09 1,3,5,6,7,8,9,14,16,17,35 template.com 
# horton-atomdb.py input g09 1 template.com 

module purge
module load Gaussian/g16
#  Step 2: run script on HPC
./run_g09.sh

## Step 3
module purge
module load horton/2.1.1
horton-atomdb.py convert --grid "exp:2e-4:20:200:266"
