#!/usr/bin/env bash

python comparison_performance.py --model=vqvae --weights=049__04-08-2020_vqvae_snn_10000_epochs_nh_256_ny_10_nframes_80 --classifier=snn --classifier_weights=002__18-06-2020_mnist_dvs_10_binary_binary_20000_epochs_nh_256_nout_128

read wait