#!/bin/bash
#SBATCH --ntasks=16
#SBATCH --time=168:00:00
#SBATCH --mem-per-cpu=8GB
#SBATCH --partition=nms_research

python /users/k1804053/snn_private/launch_experiment.py --where=rosalind --dataset=mnist_dvs_10_binary --n_h=338 --n_output_enc=338 --num_samples_train=30000 \
--test_period=2000 --num_ite=1 --suffix=_snr_m_4 \
--model=wispike --systematic=false --snr=0 --r=0.3 --labels 1 7  \
--save_path=/users/k1804053/results/025__15-07-2020_mnist_dvs_10_binary_wispike_30000_epochs_nh_338_nout_338_snr_0/

#python /users/k1804053/snn/launch_experiment.py --where=rosalind --dataset=mnist_dvs_10 --n_h=128 --num_samples_train=100000 --test_period=5000 --num_ite=1 --suffix=_teacher_forcing \
#--model=wta
#python launch_experiment.py --where=gcloud --dataset=mnist_dvs_10_binary --n_h=128 --n_output_enc=64 --num_samples_train=100000 \
#--model=wispike --systematic=true --snr=5 --r=0.3 --labels 1 7 \
#--test_period=5000 --num_ite=1 --suffix=_sytematic_snr_5_2_labels
