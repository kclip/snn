#!/usr/bin/env bash
#SBATCH --ntasks=12
#SBATCH --time=168:00:00
#SBATCH --mem-per-cpu=8GB
#SBATCH --partition=nms_research

python comparison_performance.py --where=rosalind \
 --classifier_weights=002__18-06-2020_mnist_dvs_10_binary_binary_20000_epochs_nh_256_nout_128 \
  --model=ook_ldpc --ldpc_rate=2 --snr_list 0 -1 -2 -3 -4 -5 -6 -7 -8
