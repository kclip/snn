import torch
from models.SNN import SNNetwork
from utils import utils_snn as misc_snn
from wispike.test.testing_utils import get_acc_classifier
import numpy as np
import pickle
from wispike.utils.misc import get_intermediate_dims
from wispike.utils.training_utils import init_vqvae, init_ldpc
from wispike.models.mlp import MLP


def vqvae_test(args):
    args.residual = 80 % args.n_frames
    if args.residual:
        args.n_frames += 1

    args.lr_vqvae = 0

    ### Encoder & classifier
    vqvae, _ = init_vqvae(args, args.dataset)

    weights = args.results + args.weights
    if args.classifier == 'snn':
        network = SNNetwork(**misc_snn.make_network_parameters(args.n_input_neurons,
                                                               args.n_output_neurons,
                                                               args.n_h),
                            device=args.device)
        if args.classifier_weights is not None:
            try:
                network_weights = args.results + args.classifier_weights + r'/snn_weights.hdf5'
                network.import_weights(network_weights)

            except OSError:
                network_weights = args.results + args.classifier_weights + r'/network_weights.hdf5'
                network.import_weights(network_weights)


        else:
            try:
                network_weights = weights + r'/snn_weights.hdf5'
                network.import_weights(network_weights)

            except OSError:
                network_weights = weights + r'/network_weights.hdf5'
                network.import_weights(network_weights)

        network.set_mode('test')

    elif args.classifier == 'mlp':
        n_input_neurons = np.prod(args.dataset.root.stats.train_data[1:])
        n_output_neurons = args.dataset.root.stats.train_label[1]

        network = MLP(n_input_neurons, args.n_h, n_output_neurons)

        if args.classifier_weights is not None:
            network_weights = args.results + args.classifier_weights + r'/mlp_weights.pt'
        else:
            network_weights = weights + r'/mlp_weights.pt'

        network.load_state_dict(torch.load(network_weights))

    vqvae_weights = weights + r'/vqvae_weights.pt'

    vqvae.load_state_dict(torch.load(vqvae_weights))
    vqvae.eval()

    ### Channel & coding
    args.quantized_dim, args.encodings_dim = get_intermediate_dims(vqvae, args, args.dataset)
    args.H, args.G, args.k = init_ldpc(args.encodings_dim)

    res_final = {snr: [] for snr in args.snr_list}
    res_pf = {snr: [] for snr in args.snr_list}

    for snr in args.snr_list:
        args.snr = snr

        for _ in range(args.num_ite):
            test_indices = np.random.choice(misc_snn.find_test_indices_for_labels(args.dataset, args.labels), [args.num_samples_test], replace=False)
            accs_final, accs_per_frame = get_acc_classifier(network, vqvae, args, test_indices, howto='both')

            print('snr %d, acc %f' % (snr, accs_final))
            res_final[snr].append(accs_final)
            res_pf[snr].append(accs_per_frame)

        with open(weights + r'/acc_per_snr_final_vqvae_' + args.classifier + '_nframes_%d' % args.n_frames + '.pkl', 'wb') as f:
            pickle.dump(res_final, f, pickle.HIGHEST_PROTOCOL)

        with open(weights + r'/acc_per_snr_per_frame_vqvae_' + args.classifier + '_nframes_%d' % args.n_frames + '.pkl', 'wb') as f:
            pickle.dump(res_pf, f, pickle.HIGHEST_PROTOCOL)
