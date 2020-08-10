#!/bin/bash
#SBATCH --ntasks=16
#SBATCH --time=168:00:00
#SBATCH --mem-per-cpu=8GB
#SBATCH --partition=nms_research

python /users/k1804053/snn_private/train_vqvae.py --where=rosalind --num_samples_train=1000 --test_period=100 \
--num_ite=1 --labels 1 7 --snr=-2 \
--n_frames=80