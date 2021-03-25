# snn
Code for training probabilistic binary and WTA SNNs using PyTorch.
Part of this code has been used for the following works:

N. Skatchkovsky, H. Jang, and O. Simeone, Federated Neuromorphic Learning of Spiking Neural Networks for Low-Power Edge Intelligence, accepted to IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP), 2020.
https://arxiv.org/abs/1910.09594

H. Jang, N. Skatchkovsky, and O. Simeone, VOWEL: A Local Online Learning Rule for Recurrent Networks of Probabilistic Spiking Winner-Take-All Circuits, to be presented at ICPR 2020
https://arxiv.org/abs/2004.09416

N. Skatchkovsky, H. Jang, and O. Simeone, End-to-End Learning of Neuromorphic Wireless Systems for Low-Power Edge Artificial Intelligence, to be presented at Asilomar 2020
https://arxiv.org/abs/2009.01527

# Installing 
This code can now be installed as a package and is meant to be shared in pip.
To clone and install locally the package, run 
~~~
git clone https://github.com/kclip/snn 
cd snn/ 
python -m pip install -e . 
~~~

# Run example
An experiment can be run on the MNIST-DVS dataset by launching

`python snn/launch_experiment.py`

You must first download and preprocess the MNIST-DVS dataset.

# Data preprocessing
The `data_preprocessing` module will be deprecated in following versions and is only kept for compability reasons.
Please download our `neurodata` data preprocessing and loading package instead. 
 
 
# Layered SNNs
New in the latest version: an implementation of layered SNNs, to make better use of training on GPU and train (much) larger networks in native Pytorch.
An example code is in snn.test_layered.py

 Author: Nicolas Skatchkovsky