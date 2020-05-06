#!/usr/bin/env bash
gcloud compute ssh $1 --zone europe-west1-b --command "cd /home/k1804053/snns/binary_snn && /opt/conda/bin/python mnist_online_distributed.py --dist_url='tcp://10.132.0.9:23456' \
 --world_size=$2 --node_rank=$3 --processes_per_node=$4  --dataset=/home/k1804053/mnist_dvs_binary_25ms_26pxl_10_digits.hdf5 --deltas=5 --tau=64 --num_samples=200 --num_samples_test=200 \
  --n_hidden_neurons=16 --num_ite=1 --labels 1 7"



read wait
