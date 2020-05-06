from binary_snn.models.SNN import SNNetwork
import binary_snn.utils_binary.misc as misc
from binary_snn.utils_binary.training_utils import train
from utils.filters import get_filter
import numpy as np


def launch_binary_exp(args):
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
        num_samples_train = min(args.num_samples_train, len(misc.find_test_indices_for_labels(args.dataset, args.labels)))
        indices = np.random.choice(misc.find_train_indices_for_labels(args.dataset, args.labels), [num_samples_train], replace=True)
        num_samples_test = min(args.num_samples_test, len(misc.find_test_indices_for_labels(args.dataset, args.labels)))
        test_indices = np.random.choice(misc.find_test_indices_for_labels(args.dataset, args.labels), [num_samples_test], replace=False)
    else:
        indices = np.random.choice(np.arange(args.dataset.root.stats.train_data[0]), [args.num_samples_train], replace=True)
        test_indices = np.random.choice(np.arange(args.dataset.root.stats.test_data[0]), [args.num_samples_test], replace=False)

    # Import weights if specified
    if args.weights is not None:
        network.import_weights(args.weights)

    # Start training
    test_accs = train(network, args.dataset, indices, test_indices, args.test_accs, args.lr, args.gamma, args.beta, args.kappa, args.r, args.start_idx, args.save_path)

    return test_accs
