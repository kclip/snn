#!/usr/bin/env bash
#SBATCH --ntasks=8
#SBATCH --time=168:00:00
#SBATCH --mem-per-cpu=8GB
#SBATCH --partition=nms_research

python comparison_performance.py --where=rosalind \
 --weights=054__06-08-2020_mnist_dvs_10_binary_wispike_50000_epochs_nh_676_nout_676_snr_0 \
  --model=wispike --snr_list 0

read wait