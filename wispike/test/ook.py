import torch
from models.SNN import SNNetwork
from utils import utils_snn as misc_snn
import numpy as np
import pickle
from wispike.utils.misc import channel
from wispike.test.testing_utils import classify
import pyldpc
from wispike.utils.misc import example_to_framed, channel_coding_decoding


def ook_test(args):
    args.residual = 0 #todo

    network = SNNetwork(**misc_snn.make_network_parameters(args.n_input_neurons,
                                                           args.n_output_neurons,
                                                           args.n_h),
                        device=args.device)

    weights = args.results + args.classifier_weights

    network.import_weights(weights + r'/network_weights.hdf5')

    res_final = {snr: [] for snr in args.snr_list}
    res_pf = {snr: [] for snr in args.snr_list}

    for snr in args.snr_list:
        for _ in range(args.num_ite):
            test_indices = np.random.choice(misc_snn.find_test_indices_for_labels(args.dataset, args.labels), [args.num_samples_test], replace=False)

            predictions_final = torch.zeros([args.num_samples_test], dtype=torch.long)
            predictions_pf = torch.zeros([args.num_samples_test, args.n_frames], dtype=torch.long)

            for i, idx in enumerate(test_indices):
                sample = channel(torch.FloatTensor(args.dataset.root.test.data[idx]).to(network.device), network.device, snr)
                predictions_final[i], predictions_pf[i] = classify(network, sample, args, 'both')

            true_classes = torch.max(torch.sum(torch.FloatTensor(args.dataset.root.test.label[:][test_indices]), dim=-1), dim=-1).indices

            accs_final = float(torch.sum(predictions_final == true_classes, dtype=torch.float) / len(predictions_final))
            accs_pf = torch.zeros([args.n_frames], dtype=torch.float)

            for i in range(args.n_frames):
                acc = float(torch.sum(predictions_pf[:, i] == true_classes, dtype=torch.float) / len(predictions_pf))
                accs_pf[i] = acc

            print('snr %d, acc %f' % (snr, accs_final))
            res_final[snr].append(accs_final)
            res_pf[snr].append(accs_pf)

    with open(weights + r'/acc_per_snr_final_ook.pkl', 'wb') as f:
        pickle.dump(res_final, f, pickle.HIGHEST_PROTOCOL)

    with open(weights + r'/acc_per_snr_per_frame_ook.pkl', 'wb') as f:
        pickle.dump(res_pf, f, pickle.HIGHEST_PROTOCOL)


def ook_ldpc_test(args):
    args.residual = 0  # todo
    args.n_frames = 80

    network = SNNetwork(**misc_snn.make_network_parameters(args.n_input_neurons,
                                                           args.n_output_neurons,
                                                           args.n_h),
                        device=args.device)

    weights = args.results + args.classifier_weights
    network.import_weights(weights + r'/network_weights.hdf5')

    example_frame = example_to_framed(args.dataset.root.train.data[0], args)[0].unsqueeze(0)
    frame_shape = example_frame.shape
    ldpc_codewords_length = int(args.ldpc_rate * np.prod(frame_shape))

    if args.ldpc_rate == 1.5:
        d_v = 2
        d_c = 6
    elif args.ldpc_rate == 2:
        d_v = 2
        d_c = 4
    elif args.ldpc_rate == 3:
        d_v = 2
        d_c = 3
    elif args.ldpc_rate == 4:
        d_v = 3
        d_c = 4
    elif args.ldpc_rate == 5:
        d_v = 4
        d_c = 5

    ldpc_codewords_length += d_c - (ldpc_codewords_length % d_c)

    # Make LDPC
    args.H, args.G = pyldpc.make_ldpc(ldpc_codewords_length, d_v, d_c, systematic=True, sparse=True)
    args.n, args.k = args.G.shape

    assert args.k >= np.prod(frame_shape)

    res_final = {snr: [] for snr in args.snr_list}
    res_pf = {snr: [] for snr in args.snr_list}

    for snr in args.snr_list:
        args.snr = snr
        for _ in range(args.num_ite):
            test_indices = np.random.choice(misc_snn.find_test_indices_for_labels(args.dataset, args.labels), [args.num_samples_test], replace=False)

            predictions_final = torch.zeros([args.num_samples_test], dtype=torch.long)
            predictions_pf = torch.zeros([args.num_samples_test, args.n_frames], dtype=torch.long)

            for i, idx in enumerate(test_indices):
                data = example_to_framed(args.dataset.root.test.data[idx, :, :], args)
                data_reconstructed = torch.zeros(data.shape)

                for j in range(args.n_frames):
                    frame = data[j].unsqueeze(0)
                    data_reconstructed[j] = torch.FloatTensor(channel_coding_decoding(args, frame))

                predictions_final[i], predictions_pf[i] = classify(network, data_reconstructed, args, 'both')

            true_classes = torch.max(torch.sum(torch.FloatTensor(args.dataset.root.test.label[:][test_indices]), dim=-1), dim=-1).indices

            accs_final = float(torch.sum(predictions_final == true_classes, dtype=torch.float) / len(predictions_final))
            accs_pf = torch.zeros([args.n_frames], dtype=torch.float)

            for i in range(args.n_frames):
                acc = float(torch.sum(predictions_pf[:, i] == true_classes, dtype=torch.float) / len(predictions_pf))
                accs_pf[i] = acc

            print('snr %d, acc %f' % (snr, accs_final))
            res_final[snr].append(accs_final)
            res_pf[snr].append(accs_pf)

    with open(weights + r'/acc_per_snr_final_ook_ldpc_r_%3f.pkl' % args.ldpc_rate, 'wb') as f:
        pickle.dump(res_final, f, pickle.HIGHEST_PROTOCOL)

    with open(weights + r'/acc_per_snr_per_frame_ook_ldpc_r_%3f.pkl' % args.ldpc_rate, 'wb') as f:
        pickle.dump(res_pf, f, pickle.HIGHEST_PROTOCOL)
