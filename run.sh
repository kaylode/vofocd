#!/bin/bash
#SBATCH -o %j.out
#SBATCH --gres=gpu:1
#SBATCH --nodelist=selab4
#SBATCH --mem-per-cpu=4GB

eval "$(conda shell.bash hook)"
conda activate vofo

cd /home/htluc/vocal-folds

bash /home/htluc/vocal-folds/scripts/detection/train_fasterrcnn.sh