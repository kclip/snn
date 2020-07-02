#!/bin/bash
#SBATCH --ntasks=16
#SBATCH --time=168:00:00
#SBATCH --mem-per-cpu=8GB
#SBATCH --partition=nms_research

python /users/k1804053/snn_private/train_vqvae.py --where=rosalind --n_h=512 --num_samples_train=100000 --test_period=5000 --num_ite=1 --suffix=_snr_0 \
 --snr=0