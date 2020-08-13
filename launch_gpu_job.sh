#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --mem=12GB
#SBATCH --partition=nms_research_gpu
#SBATCH --gres=gpu
#SBATCH --constrain=v100

python /users/k1804053/snn_private/launch_experiment.py --where=rosalind --dataset=mnist_dvs_10_binary \
--n_h=338 --n_output_enc=338 --num_samples_train=50000 \
--test_period=5000 --num_ite=3 --suffix=_snr_0 \
--model=wispike --systematic=false --snr=0 --r=0.3 --labels 1 7 --disable-cuda=false --rand_snr=true

#python /users/k1804053/snn_private/launch_experiment.py --where=rosalind --dataset=mnist_dvs_10_binary \
#--n_h=256 --num_samples_train=50000 \
#--test_period=5000 --num_ite=3  \
#--r=0.3 --labels 1 7 --disable-cuda=false
