#!/bin/bash
python test_performance.py --dataset=mnist_dvs_10_binary --n_h=256 --n_output_enc=256 \
--model=wispike --systematic=false --snr=-5 \
--weights=005__19-06-2020_mnist_dvs_10_binary_wispike_30000_epochs_nh_256_nout_256_snr_m_5_lr_1em4 \
--labels 1 7