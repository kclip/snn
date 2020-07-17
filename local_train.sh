#!/usr/bin/env bash
gcloud compute ssh --zone europe-west1-b $1 \
--command "cd /home/k1804053/snn_private && /opt/conda/bin/python mnist_online_distributed.py --dist_url='tcp://10.132.0.9:23456' \
 --world_size=$2 --node_rank=$3 --processes_per_node=$4 --tau=32 --test_interval=16 --num_samples_train=4500 \
  --n_hidden_neurons=64 --num_ite=3 --save_path=/home/k1804053/results/results_flsnn/"



read wait