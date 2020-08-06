#!/usr/bin/env bash

python comparison_performance.py --model=vqvae --weights=049__04-08-2020_vqvae_snn_10000_epochs_nh_256_ny_10_nframes_80 --classifier=mlp --classifier_weights=001__04-08-2020_vqvae_mlp_9000_epochs_nh_256_ny_720_nframes_80

read wait