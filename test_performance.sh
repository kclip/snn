#!/usr/bin/env bash
#SBATCH --ntasks=16
#SBATCH --time=168:00:00
#SBATCH --mem-per-cpu=8GB
#SBATCH --partition=nms_research

python comparison_performance.py --where=rosalind \
 --weights=050__04-08-2020_mnist_dvs_10_binary_wispike_50000_epochs_nh_1014_nout_1014_snr_m_6 \
  --model=wispike --snr_list 0 -6

read wait