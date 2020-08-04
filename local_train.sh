#!/usr/bin/env bash
gcloud compute ssh --zone europe-west1-b $1 \
--command "cd /home/k1804053/snn_private && /opt/conda/bin/python mnist_online_distributed.py --dist_url='tcp://10.132.0.9:23456' \
 --world_size=$2 --node_rank=$3 --processes_per_node=$4 --tau=512 --test_interval=25 --num_samples_train=2500 \
  --n_hidden_neurons=64 --num_ite=3 --save_path=/home/k1804053/results/results_flsnn/ --labels 1 7"



read wait