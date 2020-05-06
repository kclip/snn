#!/bin/bash
#SBATCH --ntasks=16
#SBATCH --time=168:00:00
#SBATCH --mem-per-cpu=8GB
#SBATCH --partition=nms_research

python mnist_test.py --where=distant --dataset=mnist_dvs_10 --n_h=512 --num_samples=90000 --num_ite=2 --suffix=_1