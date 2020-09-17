from snn.models.SNN import BinarySNN
from snn.training_utils.snn_training import train_experiment
from snn.utils.filters import get_filter
from snn.utils.misc import make_network_parameters, get_indices


def launch_binary_exp(args):
    for _ in range(args.num_ite):
        # Generate network
        network = BinarySNN(**make_network_parameters(network_type=args.model,
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


        # Import weights if resuming training
        if args.start_idx > 0:
            network.import_weights(args.save_path + r'/network_weights.hdf5')

        # Start training
        train_experiment(network, args)
