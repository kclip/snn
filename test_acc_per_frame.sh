#!/bin/bash
python comparison_performance.py --model=vqvae --n_h=256 --n_output_enc=512 \
--systematic=false --snr=0 --classifier=mlp \
--weights=018__06-07-2020_vqvae_snn_100000_epochs_nh_512_nout_360_snr_0