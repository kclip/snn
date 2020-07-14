#!/bin/bash
#SBATCH --ntasks=24
#SBATCH --time=168:00:00
#SBATCH --mem-per-cpu=8GB
#SBATCH --partition=nms_research

python /users/k1804053/snn_private/launch_experiment.py --where=rosalind --dataset=mnist_dvs_10_binary --n_h=512 --n_output_enc=512 --num_samples_train=60000 \
--test_period=2000 --num_ite=1 --suffix=_snr_m_4 \
--model=wispike --systematic=false --snr=-4 --r=0.3 --labels 1 7

#--save_path=/users/k1804053/results/009__23-06-2020_mnist_dvs_10_binary_wispike_30000_epochs_nh_512_nout_512_snr_m_6/

#python /users/k1804053/snn/launch_experiment.py --where=rosalind --dataset=mnist_dvs_10 --n_h=128 --num_samples_train=100000 --test_period=5000 --num_ite=1 --suffix=_teacher_forcing \
#--model=wta
#python launch_experiment.py --where=gcloud --dataset=mnist_dvs_10_binary --n_h=128 --n_output_enc=64 --num_samples_train=100000 \
#--model=wispike --systematic=true --snr=5 --r=0.3 --labels 1 7 \
#--test_period=5000 --num_ite=1 --suffix=_sytematic_snr_5_2_labels
