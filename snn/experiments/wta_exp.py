from snn.models.WTA_SNN import WTASNN
from snn.training_utils.multivalued_training import train
from snn.utils.filters import get_filter
from snn.utils.misc import get_indices, make_network_parameters


def launch_multivalued_exp(args):
    for _ in range(args.num_ite):
        network = WTASNN(**make_network_parameters(network_type=args.model,
                                                   n_input_neurons=args.n_input_neurons,
                                                   n_output_neurons=args.n_output_neurons,
                                                   n_hidden_neurons=args.n_hidden_neurons,
                                                   topology_type=args.topology_type,
                                                   topology=args.topology,
                                                   n_neurons_per_layer=args.n_neurons_per_layer,
                                                   density=args.density,
                                                   weights_magnitude=args.weights_magnitude,
                                                   initialization=args.initialization,
                                                   synaptic_filter=get_filter(args.syn_filter),
                                                   n_basis_ff=args.n_basis_ff,
                                                   n_basis_fb=args.n_basis_fb,
                                                   tau_ff=args.tau_ff,
                                                   tau_fb=args.tau_fb,
                                                   mu=args.mu
                                                   ),
                         device=args.device)

        # Select training and test examples from subset of labels if specified
        indices, test_indices = get_indices(args)

        # Import weights if specified
        if args.start_idx > 0:
            network.import_weights(args.save_path + r'/network_weights.hdf5')

        # Start training
        args.test_accs = train(network, args.dataset, args.sample_length, args.dt, args.input_shape, args.polarity, indices, test_indices,
                               args.lr, args.n_classes, args.r, args.beta, args.gamma, args.kappa, args.start_idx, args.test_accs, args.save_path)
