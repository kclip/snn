from wispike.utils import misc as misc_wispike
from wispike.utils.training_utils import init_training_wispike
from wispike.test.testing_utils import get_acc_wispike
from utils import utils_snn as misc_snn
from models.SNN import SNNetwork
from training_utils.snn_training import local_feedback_and_update
import torch
import numpy as np
import pickle
from utils.filters import get_filter
import os


def wispike(args):
    ### Network parameters
    n_hidden_enc = args.n_h

    if args.systematic:
        n_transmitted = args.n_input_neurons + args.n_output_enc
    else:
        n_transmitted = args.n_output_enc

    n_inputs_dec = n_transmitted
    n_hidden_dec = args.n_h

    args.lr = args.lr / (n_hidden_enc + n_hidden_dec)

    for _ in range(args.num_ite):
        ### Find indices
        if args.labels is not None:
            print(args.labels)
            indices = np.random.choice(misc_snn.find_train_indices_for_labels(args.dataset, args.labels), [args.num_samples_train], replace=True)
            num_samples_test = min(args.num_samples_test, len(misc_snn.find_test_indices_for_labels(args.dataset, args.labels)))
            test_indices = np.random.choice(misc_snn.find_test_indices_for_labels(args.dataset, args.labels), [num_samples_test], replace=False)
        else:
            indices = np.random.choice(np.arange(args.dataset.root.stats.train_data[0]), [args.num_samples_train], replace=True)
            test_indices = np.random.choice(np.arange(args.dataset.root.stats.test_data[0]), [args.num_samples_test], replace=False)

        encoder = SNNetwork(**misc_snn.make_network_parameters(args.n_input_neurons,
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
                                                               args.n_output_neurons,
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

        if args.start_idx > 0:
            encoder.import_weights(args.save_path + r'encoder_weights.hdf5')
            decoder.import_weights(args.save_path + r'decoder_weights.hdf5')

        # init training
        eligibility_trace_hidden_enc, eligibility_trace_hidden_dec, eligibility_trace_output_dec, \
            learning_signal, baseline_num_enc, baseline_den_enc, baseline_num_dec, baseline_den_dec, S_prime = init_training_wispike(encoder, decoder, args)

        for j, sample_idx in enumerate(indices):
            # if (j + 1) % args.dataset.root.train.data[:].shape[0] == 0:
            #     args.lr /= 2

            if args.test_accs:
                if (j + 1) in args.test_accs:
                    acc, _ = get_acc_wispike(encoder, decoder, args, test_indices, args.n_output_enc)
                    print('test accuracy at ite %d: %f' % (int(j + 1), acc))
                    args.test_accs[int(j + 1)].append(acc)

                    if args.save_path is not None:
                        with open(args.save_path + '/test_accs.pkl', 'wb') as f:
                            pickle.dump(args.test_accs, f, pickle.HIGHEST_PROTOCOL)

                        encoder.save(args.save_path + '/encoder_weights.hdf5')
                        decoder.save(args.save_path + '/decoder_weights.hdf5')

                    encoder.set_mode('train')
                    decoder.set_mode('train')

            misc_snn.refractory_period(encoder)
            misc_snn.refractory_period(decoder)

            sample_enc = torch.FloatTensor(args.dataset.root.train.data[sample_idx]).to(encoder.device)
            output_dec = torch.FloatTensor(args.dataset.root.train.label[sample_idx]).to(decoder.device)

            if args.rand_snr:
                args.snr = np.random.choice(np.arange(0, -9, -1))

            for s in range(S_prime):
                # Feedforward sampling encoder
                log_proba_enc = encoder(sample_enc[:, s])
                proba_hidden_enc = torch.sigmoid(encoder.potential[encoder.hidden_neurons - encoder.n_input_neurons])

                if args.systematic:
                    decoder_input = misc_wispike.channel(torch.cat((sample_enc[:, s], encoder.spiking_history[encoder.hidden_neurons[-args.n_output_enc:], -1])), decoder.device, args.snr)
                else:
                    decoder_input = misc_wispike.channel(encoder.spiking_history[encoder.hidden_neurons[-args.n_output_enc:], -1], decoder.device, args.snr)

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
                                                learning_signal, baseline_num_dec, baseline_den_dec, args.lr, args)

                eligibility_trace_hidden_enc, _, _, baseline_num_enc, baseline_den_enc \
                    = local_feedback_and_update(encoder, 0, eligibility_trace_hidden_enc, None,
                                                learning_signal, baseline_num_enc, baseline_den_enc, args.lr, args)

            if j % max(1, int(len(indices) / 5)) == 0:
                print('Step %d out of %d' % (j, len(indices)))

        # At the end of training, save final weights if none exist or if this ite was better than all the others
        if not os.path.exists(args.save_path + '/encoder_weights_final.hdf5'):
            encoder.save(args.save_path + '/encoder_weights_final.hdf5')
            decoder.save(args.save_path + '/decoder_weights_final.hdf5')
        else:
            if args.test_accs[list(args.test_accs.keys())[-1]][-1] >= max(args.test_accs[list(args.test_accs.keys())[-1]][:-1]):
                encoder.save(args.save_path + '/encoder_weights_final.hdf5')
                decoder.save(args.save_path + '/decoder_weights_final.hdf5')

