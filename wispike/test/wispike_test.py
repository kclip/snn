import torch
from multivalued_snn.utils_multivalued.misc import str2bool
from binary_snn.models.SNN import SNNetwork
from binary_snn.utils_binary import misc as misc_snn
from wispike.test.testing_utils import get_acc_wispike
import numpy as np


def wispike_test(args):
    n_hidden_enc = args.n_h

    args.systematic = str2bool(args.systematic)
    if args.systematic:
        n_transmitted = args.n_input_neurons + args.n_output_enc
    else:
        n_transmitted = args.n_output_enc

    n_inputs_dec = n_transmitted
    n_hidden_dec = args.n_h

    encoder = SNNetwork(**misc_snn.make_network_parameters(args.n_input_neurons,
                                                           0,
                                                           n_hidden_enc),
                        device=args.device)

    decoder = SNNetwork(**misc_snn.make_network_parameters(n_inputs_dec,
                                                           args.n_output_neurons,
                                                           n_hidden_dec),
                        device=args.device)

    weights = args.home + r'/results/results_wispike/' + args.weights
    encoder_weights = weights + r'/encoder_weights_final.hdf5'
    decoder_weights = weights + r'/decoder_weights_final.hdf5'

    encoder.import_weights(encoder_weights)
    decoder.import_weights(decoder_weights)

    acc_per_frame = torch.Tensor()
    for _ in range(args.num_ite):
        test_indices = np.random.choice(misc_snn.find_test_indices_for_labels(args.dataset, args.labels), [args.num_samples_test], replace=False)
        acc_per_frame = torch.cat((acc_per_frame,
                                   get_acc_wispike(encoder, decoder, args, test_indices, args.n_output_neurons, howto='per_frame').unsqueeze(1)), dim=1)
    np.save(weights + r'/acc_per_frame.npy', acc_per_frame.numpy())
