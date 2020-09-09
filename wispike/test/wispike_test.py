from utils.utils_wtasnn import str2bool
from models.SNN import SNNetwork
from utils import utils_snn as misc_snn
from wispike.test.testing_utils import get_acc_wispike
import numpy as np
import pickle


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

    weights = args.results + args.weights
    encoder_weights = weights + r'/encoder_weights_final.hdf5'
    decoder_weights = weights + r'/decoder_weights_final.hdf5'

    encoder.import_weights(encoder_weights)
    decoder.import_weights(decoder_weights)

    res_final = {snr: [] for snr in args.snr_list}
    res_pf = {snr: [] for snr in args.snr_list}

    for _ in range(args.num_ite):
        for snr in args.snr_list:
            args.snr = snr

            test_indices = np.random.choice(misc_snn.find_test_indices_for_labels(args.dataset, args.labels), [args.num_samples_test], replace=False)
            accs_final, accs_pf = get_acc_wispike(encoder, decoder, args, test_indices, args.n_output_neurons, howto='both')

            print('snr %d, acc %f' % (snr, accs_final))
            res_final[snr].append(accs_final)
            res_pf[snr].append(accs_pf)

    with open(weights + r'/acc_per_snr_final.pkl', 'wb') as f:
        pickle.dump(res_final, f, pickle.HIGHEST_PROTOCOL)

    with open(weights + r'/acc_per_snr_per_frame.pkl', 'wb') as f:
        pickle.dump(res_pf, f, pickle.HIGHEST_PROTOCOL)
