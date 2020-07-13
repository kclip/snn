#!/usr/bin/env bash
gcloud compute ssh $1 --command "cd /home/k1804053/snn && /opt/anaconda3/bin/python binary_snn/mnist_online_distributed_vanilla.py --dist_url='tcp://10.132.0.9:23456' \
 --world_size=$2 --node_rank=$3 --processes_per_node=$4 --tau=64 --test_interval=100 --num_samples_train=4500 \
  --n_hidden_neurons=64 --num_ite=3 --save_paths=/home/k1804053/results/results_flsnn/"



read wait