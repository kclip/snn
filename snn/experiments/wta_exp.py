from snn.models.WTA_SNN import WTASNN
from snn.training_utils.multivalued_training import train
from snn.utils.filters import get_filter
from snn.utils.misc import get_indices, make_network_parameters


def launch_multivalued_exp(args, params):
    for _ in range(params['num_ite']):
        network = WTASNN(**make_network_parameters(network_type=params['model'],
                                                   n_input_neurons=params['n_input_neurons'],
                                                   n_output_neurons=params['n_output_neurons'],
                                                   n_hidden_neurons=params['n_hidden_neurons'],
                                                   topology_type=params['topology_type'],
                                                   topology=params['topology'],
                                                   n_neurons_per_layer=params['n_neurons_per_layer'],
                                                   density=params['density'],
                                                   weights_magnitude=params['weights_magnitude'],
                                                   initialization=params['initialization'],
                                                   synaptic_filter=get_filter(params['syn_filter']),
                                                   n_basis_ff=params['n_basis_ff'],
                                                   n_basis_fb=params['n_basis_fb'],
                                                   tau_ff=params['tau_ff'],
                                                   tau_fb=params['tau_fb'],
                                                   mu=params['mu']
                                                   ),
                         device=args.device)

        # Import weights if specified
        if params['start_idx'] > 0:
            network.import_weights(args.save_path + r'/network_weights.hdf5')

        # Start training
        params['test_accs'] = train(network, params['dataset'], params['sample_length'], params['dt'], params['input_shape'], params['polarity'],
                                    params['indices'], params['test_indices'], params['lr'], params['n_classes'], params['pattern'], params['r'], params['beta'], params['gamma'],
                                    params['kappa'], params['start_idx'], params['test_accs'], params['save_path'])
