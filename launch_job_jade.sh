#!/bin/bash
#SBATCH --ntasks=16
#SBATCH --time=168:00:00
#SBATCH --mem-per-cpu=8GB
#SBATCH --partition=small

python /users/k1804053/snn/launch_experiment.py --where=distant --dataset=mnist_dvs_10_binary --n_h=64 --num_samples_train=100000 --test_period=5000 --num_ite=1 --suffix=_sytematic_snr_1 \
--model=wispike --systematic=true --snr=1 --r=0.3