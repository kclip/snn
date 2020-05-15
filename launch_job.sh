#!/bin/bash
#SBATCH --ntasks=16
#SBATCH --time=168:00:00
#SBATCH --mem-per-cpu=8GB
#SBATCH --partition=nms_research

python /users/k1804053/snn/launch_experiment.py --where=distant --dataset=mnist_dvs_10_binary --n_h=64 --num_samples_train=200000 --test_period=5000 --num_ite=1 --suffix=_1 --topology_type=custom