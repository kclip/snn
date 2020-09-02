from binary_snn.models.SNN import SNNetwork
import binary_snn.utils_binary.misc as misc
from binary_snn.utils_binary.training_utils import train
from utils.filters import get_filter
import numpy as np


def launch_binary_exp(args):
    for _ in range(args.num_ite):
        # Generate network
        network = SNNetwork(**misc.make_network_parameters(args.n_input_neurons,
                                                           args.n_output_neurons,
                                                           args.n_hidden_neurons,
                                                           args.topology_type,
                                                           args.topology,
                                                           args.density,
                                                           'train',
                                                           args.weights_magnitude,
                                                           args.n_basis_ff,
                                                           get_filter(args.ff_filter),
                                                           args.n_basis_fb,
                                                           get_filter(args.fb_filter),
                                                           args.initialization,
                                                           args.tau_ff,
                                                           args.tau_fb,
                                                           args.mu,
                                                           args.save_path),
                            device=args.device)

        # Select training and test examples from subset of labels if specified
        if args.labels is not None:
            print(args.labels)
            indices = np.random.choice(misc.find_indices_for_labels(args.dataset.root.train, args.labels), [args.num_samples_train], replace=True)
            args.num_samples_test = min(args.num_samples_test, len(misc.find_indices_for_labels(args.dataset.root.test, args.labels)))
            test_indices = np.random.choice(misc.find_indices_for_labels(args.dataset.root.test, args.labels), [args.num_samples_test], replace=False)
        else:
            indices = np.random.choice(np.arange(args.dataset.root.stats.train_data[0]), [args.num_samples_train], replace=True) # todo
            test_indices = np.random.choice(np.arange(args.dataset.root.stats.test_data[0]), [args.num_samples_test], replace=False)

        # Import weights if resuming training
        if args.start_idx > 0:
            network.import_weights(args.save_path + r'/network_weights.hdf5')

        # Start training
        train(network, indices, test_indices, args)
