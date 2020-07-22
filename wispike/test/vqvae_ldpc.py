import torch
from binary_snn.models.SNN import SNNetwork
from binary_snn.utils_binary import misc as misc_snn
from wispike.test.testing_utils import get_acc_classifier
import numpy as np
import pickle
from wispike.utils.misc import get_intermediate_dims
from wispike.utils.training_utils import init_vqvae, init_ldpc
from wispike.models.mlp import MLP


def vqvae_test(args):
    args.n_frames = 80

    args.residual = 80 % args.n_frames
    if args.residual:
        args.n_frames += 1

    ### Encoder & classifier
    vqvae, _ = init_vqvae(args)

    weights = args.home + r'/results/results_wispike/' + args.weights
    if args.classifier == 'snn':
        network = SNNetwork(**misc_snn.make_network_parameters(args.n_input_neurons,
                                                               args.n_output_neurons,
                                                               args.n_h),
                            device=args.device)
        network_weights = weights + r'/snn_weights.hdf5'
        network.import_weights(network_weights)
        network.set_mode('test')

    elif args.classifier == 'mlp':
        n_input_neurons = np.prod(args.dataset.root.stats.train_data[1:])
        n_output_neurons = args.dataset.root.stats.train_label[1]

        network = MLP(n_input_neurons, args.n_h, n_output_neurons)
        network_weights = weights + r'/mlp_weights.pt'
        network.load_state_dict(torch.load(network_weights))

    vqvae_weights = weights + r'/vqvae_weights.pt'

    vqvae.load_state_dict(torch.load(vqvae_weights))
    vqvae.eval()

    ### Channel & coding
    args.quantized_dim, args.encodings_dim = get_intermediate_dims(vqvae, args)
    args.H, args.G, args.k = init_ldpc(args.encodings_dim)

    snr_list = [0, -2, -4, -6, -8, -10]

    res_final = {snr: [] for snr in snr_list}
    res_pf = {snr: [] for snr in snr_list}

    for snr in snr_list:
        args.snr = snr

        for _ in range(args.num_ite):
            test_indices = np.random.choice(misc_snn.find_test_indices_for_labels(args.dataset, args.labels), [args.num_samples_test], replace=False)
            accs_final, accs_per_frame = get_acc_classifier(network, vqvae, args, test_indices, howto='both')

            print('snr %d, acc %f' % (snr, accs_final))
            res_final[snr].append(accs_final)
            res_pf[snr].append(accs_per_frame)

        with open(weights + r'/acc_per_snr_final.pkl', 'wb') as f:
            pickle.dump(res_final, f, pickle.HIGHEST_PROTOCOL)

        with open(weights + r'/acc_per_snr_per_frame.pkl', 'wb') as f:
            pickle.dump(res_pf, f, pickle.HIGHEST_PROTOCOL)
