#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --mem=12GB
#SBATCH --partition=nms_research_gpu
#SBATCH --gres=gpu
#SBATCH --constrain=v100

python /users/k1804053/snn/multivalued_snn/multivalued_test.py --where=distant --dataset=mnist_dvs_10 --mode=train_ml_online --n_h=16 --num_samples=1000000 --disable-cuda=0 \
  --suffix=_20_basis --tau_ff=60 --tau_fb=60 --mu=2.2 --n_basis_ff=20
  #--weights=/users/k1804053/snn/multivalued_snn/results/dvs_gesture_1ms_train_ml_online135000_epochs_nh_2048_1_weights.hdf5 --start_idx=9000