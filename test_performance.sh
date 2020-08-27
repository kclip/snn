#!/usr/bin/env bash
#SBATCH --ntasks=12
#SBATCH --time=168:00:00
#SBATCH --mem-per-cpu=8GB
#SBATCH --partition=nms_research

python comparison_performance.py --where=rosalind \
 --weights=067__27-08-2020_vqvae_mlp_1000_epochs_nh_256_ny_720_nframes_5 \
  --model=vqvae --n_frames=5 --snr_list 0 -6
