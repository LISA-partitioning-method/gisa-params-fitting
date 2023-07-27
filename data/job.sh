#!/bin/bash

#PBS -N GISA-proatom-dens
#PBS -l nodes=1:ppn=12
#PBS -l walltime=00:20:00
#PBS -M Yingxing.Cheng@ugent.be
#PBS -l mem=96gb


cd \$PBS_O_WORKDIR
bash main.sh

