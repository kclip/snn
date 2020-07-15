import torch
import tables
import argparse
from multivalued_snn.utils_multivalued.misc import str2bool
from binary_snn.models.SNN import SNNetwork
from binary_snn.utils_binary import misc as misc_snn
from wispike.utils.testing_utils import get_acc_wispike, classify
import numpy as np
import pickle
from binary_snn.utils_binary.misc import refractory_period
from wispike.utils.misc import channel, get_intermediate_dims, example_to_framed, channel_coding_decoding, binarize
from wispike.utils.training_utils import init_vqvae, init_ldpc

if __name__ == "__main__":
    # setting the hyper parameters
    parser = argparse.ArgumentParser(description='Train probabilistic multivalued SNNs using Pytorch')

    # Training arguments
    parser.add_argument('--where', default='local')
    parser.add_argument('--model', choices=['wispike', 'ook', 'vqvae'])
    parser.add_argument('--dataset', default='mnist_dvs_10_binary')
    parser.add_argument('--weights', type=str, default=None, help='Path to weights to load')
    parser.add_argument('--disable-cuda', type=str, default='true', help='Disable CUDA')
    parser.add_argument('--classifier', type=str, default='snn', choices=['snn', 'mlp'])

    parser.add_argument('--embedding_dim', default=32, type=int, help='Size of VQ-VAE latent embeddings')
    parser.add_argument('--num_embeddings', default=10, type=int, help='Number of VQ-VAE latent embeddings')
    parser.add_argument('--lr_vqvae', default=1e-3, type=float, help='Learning rate of VQ-VAE')
    parser.add_argument('--maxiter', default=100, type=int, help='Max number of iteration for BP decoding of LDPC code')

    # Arguments common to all models
    parser.add_argument('--n_h', default=256, type=int, help='Number of hidden neurons')

    # Arguments for Wispike
    parser.add_argument('--systematic', type=str, default='false', help='Systematic communication')
    parser.add_argument('--snr', type=float, default=None, help='SNR')
    parser.add_argument('--n_output_enc', default=128, type=int, help='')

    args = parser.parse_args()

print(args)

if args.where == 'local':
    home = r'C:/Users/K1804053/PycharmProjects'
elif args.where == 'rosalind':
    home = r'/users/k1804053'
elif args.where == 'jade':
    home = r'/jmain01/home/JAD014/mxm09/nxs94-mxm09'
elif args.where == 'gcloud':
    home = r'/home/k1804053'


dataset = home + r'/datasets/mnist-dvs/mnist_dvs_binary_25ms_26pxl_10_digits.hdf5'

args.dataset = tables.open_file(dataset)

args.disable_cuda = str2bool(args.disable_cuda)
args.device = None
if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')

args.save_path = None

### Learning parameters
args.num_samples_test = args.dataset.root.stats.test_data[0]
args.labels = [1, 7]
num_samples_test = min(args.num_samples_test, len(misc_snn.find_test_indices_for_labels(args.dataset, args.labels)))
test_indices = np.random.choice(misc_snn.find_test_indices_for_labels(args.dataset, args.labels), [num_samples_test], replace=False)

### Network parameters
args.n_input_neurons = args.dataset.root.stats.train_data[1]
args.n_output_neurons = args.dataset.root.stats.train_label[1]
args.n_hidden_neurons = args.n_h


if args.model == 'wispike':
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

    weights = home + r'/results/results_wispike/' + args.weights
    encoder_weights = weights + r'/encoder_weights_final.hdf5'
    decoder_weights = weights + r'/decoder_weights_final.hdf5'

    encoder.import_weights(encoder_weights)
    decoder.import_weights(decoder_weights)

    acc_per_frame = get_acc_wispike(encoder, decoder, args, test_indices, args.n_output_neurons, howto='per_frame')


    np.save(weights + r'/acc_per_frame.npy', acc_per_frame.numpy())


elif args.model == 'ook':
    network = SNNetwork(**misc_snn.make_network_parameters(args.n_input_neurons,
                                                           args.n_output_neurons,
                                                           args.n_h),
                        device=args.device)

    weights = home + r'/results/results_wispike/' + args.weights
    network.import_weights(weights + r'/mlp_weights.hdf5')
    snr_list = [0]
    # network.import_weights(weights + r'/network_weights.hdf5')
    # snr_list = [0, -2, -4, -6, -8, -10]

    res_final = {snr: 0 for snr in snr_list}
    res_pf = {snr: 0 for snr in snr_list}

    for snr in snr_list:
        network.set_mode('test')
        network.reset_internal_state()

        T = args.dataset.root.test.label[:].shape[-1]

        outputs = torch.zeros([len(test_indices), network.n_output_neurons, T])
        loss = 0

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

        print('snr %d, acc %f' % (snr, acc))
        res_final[snr] = accs_final
        res_pf[snr] = accs_per_frame

    with open(weights + r'/acc_per_snr_final.pkl', 'wb') as f:
        pickle.dump(res_final, f, pickle.HIGHEST_PROTOCOL)

    with open(weights + r'/acc_per_snr_per_frame.pkl', 'wb') as f:
        pickle.dump(res_pf, f, pickle.HIGHEST_PROTOCOL)


elif args.model == 'vqvae':
    args.n_frames = 80

    args.residual = 80 % args.n_frames
    if args.residual:
        args.n_frames += 1

    ### Encoder & classifier
    network = SNNetwork(**misc_snn.make_network_parameters(args.n_input_neurons,
                                                           args.n_output_neurons,
                                                           args.n_h),
                        device=args.device)

    vqvae, _ = init_vqvae(args)

    weights = home + r'/results/results_wispike/' + args.weights
    if args.classifier == 'snn':
        network_weights = weights + r'/snn_weights.hdf5'
    elif args.classifier == 'mlp':
        network_weights = weights + r'/mlp_weights.hdf5'
    vqvae_weights = weights + r'/vqvae_weights.pt'

    network.import_weights(network_weights)
    vqvae.load_state_dict(torch.load(vqvae_weights))
    network.set_mode('test')
    vqvae.eval()

    ### Channel & coding
    args.quantized_dim, args.encodings_dim = get_intermediate_dims(vqvae, args)
    args.H, args.G, args.k = init_ldpc(args.encodings_dim)

    snr_list = [0, -2, -4, -6, -8, -10]

    res_final = {snr: 0 for snr in snr_list}
    res_pf = {snr: 0 for snr in snr_list}

    for snr in snr_list:
        args.snr = snr
        network.reset_internal_state()

        predictions_final = torch.zeros([len(test_indices)], dtype=torch.long)
        predictions_pf = torch.zeros([len(test_indices), args.n_frames], dtype=torch.long)

        T = args.dataset.root.test.label[:].shape[-1]

        outputs = torch.zeros([len(test_indices), network.n_output_neurons, T])

        for i, idx in enumerate(test_indices):
            data = example_to_framed(args.dataset.root.test.data[idx, :, :], args)
            data_reconstructed = torch.zeros(data.shape)

            for t in range(T):
                frame = data[t].unsqueeze(0)
                with torch.autograd.no_grad():
                    _, encodings = vqvae.encode(frame)
                    encodings_decoded = channel_coding_decoding(args, encodings)
                    data_reconstructed[t] = vqvae.decode(encodings_decoded, args.quantized_dim)

            predictions_final[i], predictions_pf[i] = classify(network, data_reconstructed, args, 'both')

        true_classes = torch.max(torch.sum(torch.FloatTensor(args.dataset.root.test.label[:][test_indices]), dim=-1), dim=-1).indices

        accs_final = float(torch.sum(predictions_final == true_classes, dtype=torch.float) / len(predictions_final))
        accs_per_frame = torch.zeros([T], dtype=torch.float)
        for t in range(1, T):
            acc = float(torch.sum(predictions_pf[:, t] == true_classes, dtype=torch.float) / len(predictions_pf))
            accs_per_frame[t] = acc

        print('snr %d, acc %f' % (snr, acc))
        res_final[snr] = accs_final
        res_pf[snr] = accs_per_frame

    with open(weights + r'/acc_per_snr_final.npy', 'wb') as f:
        pickle.dump(res_final, f, pickle.HIGHEST_PROTOCOL)

    with open(weights + r'/acc_per_snr_per_frame.npy', 'wb') as f:
        pickle.dump(res_pf, f, pickle.HIGHEST_PROTOCOL)
