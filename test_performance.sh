#!/usr/bin/env bash

python comparison_performance.py --model=ook_ldpc --classifier=snn --classifier_weights=002__18-06-2020_mnist_dvs_10_binary_binary_20000_epochs_nh_256_nout_128  --ldpc_rate=1.5

python comparison_performance.py --weights=055__07-08-2020_vqvae_snn_1000_epochs_nh_256_ny_720_nframes_80 --model=vqvae --classifier=snn --classifier_weights=002__18-06-2020_mnist_dvs_10_binary_binary_20000_epochs_nh_256_nout_128

read wait