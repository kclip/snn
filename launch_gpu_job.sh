#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --mem=12GB
#SBATCH --partition=nms_research_gpu
#SBATCH --gres=gpu
#SBATCH --constrain=v100

python /users/k1804053/snn_private/launch_experiment.py --where=rosalind --dataset=mnist_dvs_10_binary \
--n_h=1014 --n_output_enc=1014 --num_samples_train=50000 \
--test_period=5000 --num_ite=3 --suffix=_snr_m_6 \
--model=wispike --systematic=false --snr=-6 --r=0.3 --labels 1 7 --disable-cuda=false
