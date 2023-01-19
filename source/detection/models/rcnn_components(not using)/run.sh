#!/bin/bash
#SBATCH -o %j.out
#SBATCH --gres=gpu:1
#SBATCH --nodelist=selab4
#SBATCH --mem-per-cpu=4GB

eval "$(conda shell.bash hook)"
conda activate vofo

cd /home/htluc/vocal-folds/source/detection/mask_rcnn
python3 model.py