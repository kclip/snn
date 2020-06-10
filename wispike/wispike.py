from wispike.utils import misc as misc_wispike
from wispike.utils.training_utils import init_training_wispike
from binary_snn.utils_binary import misc as misc_snn
from binary_snn.models.SNN import SNNetwork
from binary_snn.utils_binary.training_utils import local_feedback_and_update
import torch
import numpy as np
import pickle
from utils.filters import get_filter


def wispike(args):
    ### Network parameters
    n_inputs_enc = args.dataset.root.stats.train_data[:][1]
    n_hidden_enc = args.n_h
    n_outputs_enc = args.n_output_enc

    if args.systematic:
        n_transmitted = n_inputs_enc + n_outputs_enc
    else:
        n_transmitted = n_outputs_enc

    n_inputs_dec = n_transmitted
    n_hidden_dec = args.n_h
    n_outputs_dec = args.dataset.root.stats.train_label[1]

    args.lr = args.lr / (n_hidden_enc + n_hidden_dec)

    ### Find indices
    if args.labels is not None:
        print(args.labels)
        indices = np.random.choice(misc_snn.find_train_indices_for_labels(args.dataset, args.labels), [args.num_samples_train], replace=True)
        num_samples_test = min(args.num_samples_test, len(misc_snn.find_test_indices_for_labels(args.dataset, args.labels)))
        test_indices = np.random.choice(misc_snn.find_test_indices_for_labels(args.dataset, args.labels), [num_samples_test], replace=False)
    else:
        indices = np.random.choice(np.arange(args.dataset.root.stats.train_data[0]), [args.num_samples_train], replace=True)
        test_indices = np.random.choice(np.arange(args.dataset.root.stats.test_data[0]), [args.num_samples_test], replace=False)

    encoder = SNNetwork(**misc_snn.make_network_parameters(n_inputs_enc,
                                                           0,
                                                           n_hidden_enc,
                                                           args.topology_type,
                                                           args.topology,
                                                           args.density,
                                                           'train',
                                                           args.weights_magnitude,
                                                           args.n_basis_ff,
                                                           get_filter(args.ff_filter),
                                                           args.n_basis_fb,
                                                           get_filter(args.fb_filter),
                                                           args.initialization,
                                                           args.tau_ff,
                                                           args.tau_fb,
                                                           args.mu,
                                                           args.save_path),
                        device=args.device)

    decoder = SNNetwork(**misc_snn.make_network_parameters(n_inputs_dec,
                                                           n_outputs_dec,
                                                           n_hidden_dec,
                                                           args.topology_type,
                                                           args.topology,
                                                           args.density,
                                                           'train',
                                                           args.weights_magnitude,
                                                           args.n_basis_ff,
                                                           get_filter(args.ff_filter),
                                                           args.n_basis_fb,
                                                           get_filter(args.fb_filter),
                                                           args.initialization,
                                                           args.tau_ff,
                                                           args.tau_fb,
                                                           args.mu,
                                                           args.save_path),
                        device=args.device)


    # init training
    eligibility_trace_hidden_enc, eligibility_trace_hidden_dec, eligibility_trace_output_dec, \
        learning_signal, baseline_num_enc, baseline_den_enc, baseline_num_dec, baseline_den_dec, S_prime = init_training_wispike(encoder, decoder, args)

    for j, sample_idx in enumerate(indices):
        # if (j + 1) % args.dataset.root.train.data[:].shape[0] == 0:
        #     args.lr /= 2

        if args.test_accs:
            if (j + 1) in args.test_accs:
                acc = misc_wispike.get_acc_wispike(encoder, decoder, args, test_indices, n_outputs_enc)
                print('test accuracy at ite %d: %f' % (int(j + 1), acc))
                args.test_accs[int(j + 1)].append(acc)

                if args.save_path is not None:
                    with open(args.save_path, 'wb') as f:
                        pickle.dump(args.test_accs, f, pickle.HIGHEST_PROTOCOL)


                encoder.set_mode('train')
                decoder.set_mode('train')

        misc_snn.refractory_period(encoder)
        misc_snn.refractory_period(decoder)

        sample_enc = torch.FloatTensor(args.dataset.root.train.data[sample_idx]).to(encoder.device)
        output_dec = torch.FloatTensor(args.dataset.root.train.label[sample_idx]).to(decoder.device)

        for s in range(S_prime):
            # Feedforward sampling encoder
            log_proba_enc = encoder(sample_enc[:, s])
            proba_hidden_enc = torch.sigmoid(encoder.potential[encoder.hidden_neurons - encoder.n_input_neurons])

            if args.systematic:
                decoder_input = misc_wispike.channel(torch.cat((sample_enc[:, s], encoder.spiking_history[encoder.hidden_neurons[-n_outputs_enc:], -1])), decoder.device, args.snr)
            else:
                decoder_input = misc_wispike.channel(encoder.spiking_history[encoder.hidden_neurons[-n_outputs_enc:], -1], decoder.device, args.snr)

            sample_dec = torch.cat((decoder_input, output_dec[:, s]), dim=0).to(decoder.device)

            log_proba_dec = decoder(sample_dec)
            proba_hidden_dec = torch.sigmoid(decoder.potential[decoder.hidden_neurons - decoder.n_input_neurons])

            ls = torch.sum(log_proba_dec[decoder.output_neurons - decoder.n_input_neurons]) \
                 - args.gamma * torch.sum(torch.cat((encoder.spiking_history[encoder.hidden_neurons, -1], decoder.spiking_history[decoder.hidden_neurons, -1]))
                                          * torch.log(1e-12 + torch.cat((proba_hidden_enc, proba_hidden_dec)) / args.r)
                                          + (1 - torch.cat((encoder.spiking_history[encoder.hidden_neurons, -1], decoder.spiking_history[decoder.hidden_neurons, -1])))
                                          * torch.log(1e-12 + (1. - torch.cat((proba_hidden_enc, proba_hidden_dec))) / (1 - args.r)))

            # Local feedback and update
            eligibility_trace_hidden_dec, eligibility_trace_output_dec, learning_signal, baseline_num_dec, baseline_den_dec \
                = local_feedback_and_update(decoder, ls, eligibility_trace_hidden_dec, eligibility_trace_output_dec,
                                            learning_signal, baseline_num_dec, baseline_den_dec, args.lr, args.beta, args.kappa)

            eligibility_trace_hidden_enc, _, _, baseline_num_enc, baseline_den_enc \
                = local_feedback_and_update(encoder, 0, eligibility_trace_hidden_enc, None,
                                            learning_signal, baseline_num_enc, baseline_den_enc, args.lr, args.beta, args.kappa)


        if j % max(1, int(len(indices) / 5)) == 0:
            print('Step %d out of %d' % (j, len(indices)))
