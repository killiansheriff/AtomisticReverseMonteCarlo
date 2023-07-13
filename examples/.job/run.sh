#!/bin/bash
#SBATCH --no-requeue
#SBATCH --job-name=run.sh
#SBATCH --output=run.out
#SBATCH --mem=200Gb
#SBATCH --time=20:00:00
#SBATCH --account=sua183
#SBATCH --cpus-per-task=10
#SBATCH --partition=shared
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
date 
 source activate test 
 python -u main.py  
 date 