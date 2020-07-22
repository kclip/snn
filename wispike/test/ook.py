import torch
from binary_snn.models.SNN import SNNetwork
from binary_snn.utils_binary import misc as misc_snn
import numpy as np
import pickle
from binary_snn.utils_binary.misc import refractory_period
from wispike.utils.misc import channel


def ook_test(args):
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

            network.set_mode('test')
            network.reset_internal_state()

            T = args.dataset.root.test.label[:].shape[-1]

            outputs = torch.zeros([len(test_indices), network.n_output_neurons, T])

            for j, sample_idx in enumerate(test_indices):
                refractory_period(network)

                sample = channel(torch.FloatTensor(args.dataset.root.test.data[sample_idx]).to(network.device), network.device, snr)

                for t in range(T):
                    log_proba = network(sample[:, t])
                    outputs[j, :, t] = network.spiking_history[network.output_neurons, -1]

            true_classes = torch.max(torch.sum(torch.FloatTensor(args.dataset.root.test.label[:][test_indices]), dim=-1), dim=-1).indices

            predictions_final = torch.max(torch.sum(outputs, dim=-1), dim=-1).indices
            accs_final = float(torch.sum(predictions_final == true_classes, dtype=torch.float) / len(predictions_final))

            accs_per_frame = torch.zeros([T], dtype=torch.float)
            for t in range(1, T):
                predictions_pf = torch.sum(outputs[:, :, :t], dim=-1).argmax(-1)
                acc = float(torch.sum(predictions_pf == true_classes, dtype=torch.float) / len(predictions_pf))
                accs_per_frame[t] = acc

            print('snr %d, acc %f' % (snr, accs_final))
            res_final[snr].append(accs_final)
            res_pf[snr].append(accs_per_frame)

    with open(weights + r'/acc_per_snr_final.pkl', 'wb') as f:
        pickle.dump(res_final, f, pickle.HIGHEST_PROTOCOL)

    with open(weights + r'/acc_per_snr_per_frame.pkl', 'wb') as f:
        pickle.dump(res_pf, f, pickle.HIGHEST_PROTOCOL)
