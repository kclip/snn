#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --mem=12GB
#SBATCH --partition=nms_research_gpu
#SBATCH --gres=gpu
#SBATCH --constrain=v100

python /users/k1804053/snn_private/launch_experiment.py --where=rosalind --dataset=mnist_dvs \
--n_h=1048 --num_samples_train=100000 \
--dt=20000 --sample_length=1800 \
--test_period=50000 --num_ite=1 --disable-cuda=false
