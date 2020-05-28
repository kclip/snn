#!/bin/bash
#SBATCH --ntasks=16
#SBATCH --time=168:00:00
#SBATCH --mem-per-cpu=8GB
#SBATCH --partition=nms_research

python /users/k1804053/snn/launch_experiment.py --where=rosalind --dataset=mnist_dvs_10_binary --n_h=128 --num_samples_train=100000 --test_period=5000 --num_ite=1 --suffix=_sytematic_snr_5 \
--model=wispike --systematic=true --snr=5 --r=0.3

#python /users/k1804053/snn/launch_experiment.py --where=rosalind --dataset=mnist_dvs_10 --n_h=128 --num_samples_train=100000 --test_period=5000 --num_ite=1 --suffix=_teacher_forcing \
#--model=wta