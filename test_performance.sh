#!/usr/bin/env bash
#SBATCH --ntasks=12
#SBATCH --time=168:00:00
#SBATCH --mem-per-cpu=8GB
#SBATCH --partition=nms_research

python comparison_performance.py --where=rosalind \
 --weights=058__11-08-2020_vqvae_mlp_1000_epochs_nh_256_ny_720_nframes_40 \
 --classifier_weights=001__04-08-2020_vqvae_mlp_9000_epochs_nh_256_ny_720_nframes_80 \
  --model=vqvae --n_frames=40 --snr_list 0 -1 -2 -3 -4 -5 -6 -7 -8
