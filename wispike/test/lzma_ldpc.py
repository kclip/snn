from wispike.utils.misc import *
from wispike.utils import training_utils
from wispike.test import testing_utils
import lzma
from models.SNN import SNNetwork
from utils import utils_snn as misc_snn
from wispike.models.mlp import MLP
import pickle


def lzma_test(args):
    args.residual = 80 % args.n_frames
    if args.residual:
        args.n_frames += 1

    # Make classifier
    weights = args.home + r'/results/results_wispike/' + args.weights
    if args.classifier == 'snn':
        network = SNNetwork(**misc_snn.make_network_parameters(args.n_input_neurons,
                                                               args.n_output_neurons,
                                                               args.n_h),
                            device=args.device)
        # network_weights = weights + r'/snn_weights.hdf5'
        network_weights = weights + r'/network_weights.hdf5'
        network.import_weights(network_weights)
        network.set_mode('test')

    elif args.classifier == 'mlp':
        n_input_neurons = np.prod(args.dataset.root.stats.train_data[1:])
        n_output_neurons = args.dataset.root.stats.train_label[1]

        network = MLP(n_input_neurons, args.n_h, n_output_neurons)
        network_weights = weights + r'/mlp_weights.pt'
        network.load_state_dict(torch.load(network_weights))

    predictions_final = torch.zeros([args.num_samples_test], dtype=torch.long)
    predictions_pf = torch.zeros([args.num_samples_test, args.n_frames], dtype=torch.long)

    snr_list = [5]#, -2, -4, -6, -8, -10]

    res_final = {snr: [] for snr in snr_list}
    res_pf = {snr: [] for snr in snr_list}

    for snr in snr_list:
        args.snr = snr

        for _ in range(args.num_ite):
            test_indices = np.random.choice(misc_snn.find_test_indices_for_labels(args.dataset, args.labels), [args.num_samples_test], replace=False)

            for i, idx in enumerate(test_indices):
                example = args.dataset.root.test.data[idx]
                frames = example_to_framed(example, args).numpy().reshape([args.n_frames, -1])
                compressed_frames = []

                for frame in frames:
                    frame_as_bytes = binarr2bytes(frame)
                    compressed_frame = lzma.compress(frame_as_bytes)
                    print(compressed_frame[:27])
                    compressed_frame_bin = bytes2binarr(compressed_frame)
                    compressed_frames.append(compressed_frame_bin)

                compressed_frames_shapes = [len(i) for i in compressed_frames]

                args.H, args.G, args.k = training_utils.init_ldpc(max(compressed_frames_shapes))

                received_frames = torch.zeros(frames.shape)
                for j, compressed_frame in enumerate(compressed_frames):
                    frame_received = channel_coding_decoding(args, compressed_frame)

                    # print(float(np.sum(frame_received == compressed_frame)) / np.prod(frame_received.shape))
                    # Recover the (known, never changing) header/footer of lzma compression
                    # frame_received[:27] = compressed_frame_bin[:27]
                    # frame_received[-12:] = compressed_frame_bin[-12:]

                    frame_received = binarr2bytes(frame_received.astype(bool))
                    # print(frame_received[:27])
                    # print(compressed_frames[j][:27])

                    # print(frame_received[-12:])
                    try:
                        frame_received_decompressed = lzma.decompress(frame_received)
                        frame_received_decompressed_bin = torch.tensor(bytes2binarr(frame_received_decompressed))
                        received_frames[j] = frame_received_decompressed_bin
                        print('Decoding successful, example %d frame %d' % (i, j))
                    except lzma.LZMAError:
                        print('Decoding failed, example %d frame %d' % (i, j))
                        continue

                received_example = framed_to_example(received_frames, args)
                # print(float(torch.sum(received_example == torch.FloatTensor(example))) / received_example.numel())

                predictions_final[i], predictions_pf = testing_utils.classify(network, received_example, args, 'both')
                print(predictions_final[i], torch.max(torch.sum(torch.FloatTensor(args.dataset.root.test.label[:][idx]), dim=-1), dim=-1).indices)

            true_classes = torch.max(torch.sum(torch.FloatTensor(args.dataset.root.test.label[:][test_indices]), dim=-1), dim=-1).indices

            accs_final = float(torch.sum(predictions_final == true_classes, dtype=torch.float) / len(predictions_final))
            accs_pf = torch.zeros([args.n_frames], dtype=torch.float)

            for i in range(args.n_frames):
                acc = float(torch.sum(predictions_pf[:, i] == true_classes, dtype=torch.float) / len(predictions_pf))
                accs_pf[i] = acc

            print('snr %d, acc %f' % (snr, accs_final))
            res_final[snr].append(accs_final)
            res_pf[snr].append(accs_pf)

            with open(weights + r'/acc_per_snr_final_lzma.pkl', 'wb') as f:
                pickle.dump(res_final, f, pickle.HIGHEST_PROTOCOL)

            with open(weights + r'/acc_per_snr_per_frame_lzma.pkl', 'wb') as f:
                pickle.dump(res_pf, f, pickle.HIGHEST_PROTOCOL)

