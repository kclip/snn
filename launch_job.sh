#!/bin/bash
#SBATCH --ntasks=16
#SBATCH --time=168:00:00
#SBATCH --mem-per-cpu=8GB
#SBATCH --partition=nms_research

python /users/k1804053/snn_private/launch_experiment.py --where=rosalind --dataset=mnist_dvs \
--n_h=512 --num_samples_train=100000 \
--dt=5000  \
--test_period=50000 --num_ite=1
