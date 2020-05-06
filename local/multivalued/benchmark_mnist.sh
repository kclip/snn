#!/bin/bash
#SBATCH --ntasks=16
#SBATCH --time=168:00:00
#SBATCH --mem-per-cpu=12GB
#SBATCH --partition=nms_research

#python /users/k1804053/snn/multivalued_snn/multivalued_exp.py --where=distant --dataset=dvs_gesture_20ms --mode=train_ml_online --num_samples=90000 --n_h=256 --suffix=_4_labels_3 \
#--labels 0 1 3 7
#python /users/k1804053/FL-SNN-multivalued/forgetting_experience.py --where=distant
python /users/k1804053/snn/multivalued_snn/multivalued_test.py --where=distant --dataset=mnist_dvs_10 --mode=train_ml_online --num_samples=1000000 --n_h=0 --suffix=_1 --tau_ff=60
--tau_fb=60 --mu=2.2 --n_basis_ff=20
# --weights=/users/k1804053/snn/multivalued_snn/results/mnist_dvs_10_train_ml_online90000_epochs_nh_0_2_weights.hdf5