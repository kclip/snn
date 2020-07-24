import torch
from binary_snn.models.SNN import SNNetwork
from binary_snn.utils_binary import misc as misc_snn
import numpy as np
import pickle
from binary_snn.utils_binary.misc import refractory_period
from wispike.utils.misc import channel
from wispike.test.testing_utils import classify


def ook_test(args):
    args.residual = 0 #todo

    network = SNNetwork(**misc_snn.make_network_parameters(args.n_input_neurons,
                                                           args.n_output_neurons,
                                                           args.n_h),
                        device=args.device)

    weights = args.home + r'/results/results_wispike/' + args.weights
    # network.import_weights(weights + r'/mlp_weights.hdf5')
    snr_list = [0]
    network.import_weights(weights + r'/network_weights.hdf5')
    # snr_list = [0, -2, -4, -6, -8, -10]

    res_final = {snr: [] for snr in snr_list}
    res_pf = {snr: [] for snr in snr_list}

    for snr in snr_list:
        for _ in range(3):
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
