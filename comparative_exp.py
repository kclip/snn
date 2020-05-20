import numpy as np
import tables
from wispike.vqvae_zalando import Model
import torch
import argparse
import torch.nn.functional as F
import os
import binary_snn.utils_binary.misc as misc
from multivalued_snn.utils_multivalued.misc import str2bool
import torch.optim as optim


def train(sample, model, optimizer, args):
    optimizer.zero_grad()
    x_tilde, z_e_x, z_q_x = model(sample, args.snr)

    # Reconstruction loss
    loss_recons = F.mse_loss(x_tilde, sample)
    # Vector quantization objective
    loss_vq = F.mse_loss(z_q_x, z_e_x.detach())
    # Commitment objective
    loss_commit = F.mse_loss(z_e_x, z_q_x.detach())

    loss = loss_recons + loss_vq + args.beta * loss_commit
    loss.backward()

    optimizer.step()
    args.steps += 1


def test(test_indices, model, args):
    loss_recons, loss_vq = 0., 0.

    for i, sample_idx in test_indices:
        sample = torch.cat((torch.FloatTensor(dataset.root.test.data[sample_idx]),
                            torch.FloatTensor(dataset.root.test.label[sample_idx])), dim=0).to(args.device)
        with torch.no_grad():

            x_tilde, z_e_x, z_q_x = model(sample)
            loss_recons += F.mse_loss(x_tilde, sample)
            loss_vq += F.mse_loss(z_q_x, z_e_x)

    return loss_recons.item(), loss_vq.item()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='VQ-VAE')

    # General
    parser.add_argument('--where', default='local')
    parser.add_argument('--dataset', default='mnist_dvs_10_binary')
    parser.add_argument('--snr', default=1000000)
    parser.add_argument('--disable-cuda', type=str, default='true', help='Disable CUDA')
    parser.add_argument('--labels', nargs='+', default=None, type=int, help='Class labels to be used during training')


    # Optimization
    parser.add_argument('--num_samples_train', type=int, default=50000,
        help='number of epochs (default: 100)')
    parser.add_argument('--batch_size', type=int, default=10,
        help='number of epochs (default: 100)')

    parser.add_argument('--num_samples_test', default=None, type=int, help='Number of samples to test on')

    parser.add_argument('--lr', type=float, default=2e-4,
        help='learning rate for Adam optimizer (default: 2e-4)')
    parser.add_argument('--beta', type=float, default=1.0,
        help='contribution of commitment loss, between 0.1 and 2.0 (default: 1.0)')

    args = parser.parse_args()


if args.where == 'local':
    data_path = r'C:/Users/K1804053/PycharmProjects/datasets/'
elif args.where == 'distant':
    data_path = r'/users/k1804053/datasets/'
elif args.where == 'gcloud':
    data_path = r'/home/k1804053/datasets/'

save_path = os.getcwd() + r'/results'

datasets = {'mnist_dvs_2': r'mnist_dvs_25ms_26pxl_2_digits_polarity.hdf5',
            'mnist_dvs_10_binary': r'mnist_dvs_binary_25ms_26pxl_10_digits.hdf5',
            'mnist_dvs_10': r'mnist_dvs_25ms_26pxl_10_digits_polarity.hdf5',
            'mnist_dvs_10_c_3': r'mnist_dvs_25ms_26pxl_10_digits_C_3.hdf5',
            'mnist_dvs_10_c_5': r'mnist_dvs_25ms_26pxl_10_digits_C_5.hdf5',
            'mnist_dvs_10_c_7': r'mnist_dvs_25ms_26pxl_10_digits_C_7.hdf5',
            'mnist_dvs_10ms_polarity': r'mnist_dvs_10ms_26pxl_10_digits_polarity.hdf5',
            'dvs_gesture_5ms': r'dvs_gesture_5ms_11_classes.hdf5',
            'dvs_gesture_5ms_5_classes': r'dvs_gesture_5ms_5_classes.hdf5',
            'dvs_gesture_20ms_2_classes': r'dvs_gesture_20ms_2_classes.hdf5',
            'dvs_gesture_5ms_2_classes': r'dvs_gesture_5ms_2_classes.hdf5',
            'dvs_gesture_5ms_3_classes': r'dvs_gesture_5ms_3_classes.hdf5',
            'dvs_gesture_15ms': r'dvs_gesture_15ms_11_classes.hdf5',
            'dvs_gesture_20ms': r'dvs_gesture_20ms_11_classes.hdf5',
            'dvs_gesture_30ms': r'dvs_gesture_30ms_11_classes.hdf5',
            'dvs_gesture_20ms_5_classes': r'dvs_gesture_20ms_5_classes.hdf5',
            'dvs_gesture_1ms': r'dvs_gesture_1ms_11_classes.hdf5',
            'shd_eng_c_2': r'shd_10ms_10_classes_eng_C_2.hdf5',
            'shd_all_c_2': r'shd_10ms_10_classes_all_C_2.hdf5'
            }

if args.dataset[:3] == 'shd':
    dataset = data_path + r'/shd/' + datasets[args.dataset]
elif args.dataset[:5] == 'mnist':
    dataset = data_path + r'/mnist-dvs/' + datasets[args.dataset]
elif args.dataset[:11] == 'dvs_gesture':
    dataset = data_path + r'/DvsGesture/' + datasets[args.dataset]
elif args.dataset[:7] == 'swedish':
    dataset = data_path + r'/SwedishLeaf_processed/' + datasets[args.dataset]
else:
    print('Error: dataset not found')

args.disable_cuda = str2bool(args.disable_cuda)
args.device = None
if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')

args.dataset = tables.open_file(dataset)

# Make VAE
batch_size = 80
num_input_channels = 1

num_hiddens = 128
num_residual_hiddens = 32
num_residual_layers = 2

embedding_dim = 16
num_embeddings = 512

commitment_cost = 0.25

decay = 0.99

learning_rate = 1e-3


# LDPC coding
ldpc_codewords_length = 676
d_v = 3
d_c = 4
snr = 1000000

model = Model(num_input_channels, num_hiddens, num_residual_layers, num_residual_hiddens, num_embeddings, embedding_dim, commitment_cost, ldpc_codewords_length, d_v, d_c, decay).to(args.device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)


if not args.num_samples_train:
    args.num_samples_train = args.dataset.root.stats.train_data[0]

if not args.num_samples_test:
    args.num_samples_test = args.dataset.root.stats.test_data[0]

if args.labels is not None:
    print(args.labels)
    indices = np.random.choice(misc.find_train_indices_for_labels(args.dataset, args.labels), [args.num_samples_train], replace=True)
    num_samples_test = min(args.num_samples_test, len(misc.find_test_indices_for_labels(args.dataset, args.labels)))
    test_indices = np.random.choice(misc.find_test_indices_for_labels(args.dataset, args.labels), [num_samples_test], replace=False)
else:
    # indices = np.vstack([np.random.choice(np.arange(args.dataset.root.stats.train_data[0]), [batch_size], replace=False) for _ in range(args.num_samples_train // batch_size)])
    indices = np.random.choice(np.arange(args.dataset.root.stats.train_data[0]), [args.num_samples_train], replace=True)
    test_indices = np.random.choice(np.arange(args.dataset.root.stats.test_data[0]), [args.num_samples_test], replace=False)

best_loss = -1.


model.train()
train_res_recon_error = []
train_res_perplexity = []


for i, sample_idxs in enumerate(indices):
    # data = torch.sum(torch.FloatTensor(args.dataset.root.train.data[sample_idxs, :, :]), dim=-1).reshape([batch_size, 1, 26, 26])
    data = torch.FloatTensor(args.dataset.root.train.data[sample_idxs, :, :]).transpose(1, 0).unsqueeze(0).unsqueeze(3).reshape([batch_size, 1, 26, 26])

    # print(data.shape)

    optimizer.zero_grad()

    vq_loss, data_recon, perplexity = model(data, snr)
    recon_error = F.mse_loss(data_recon, data)
    loss = recon_error + vq_loss
    loss.backward()

    optimizer.step()

    train_res_recon_error.append(recon_error.item())
    train_res_perplexity.append(perplexity.item())

    if (i + 1) % 10 == 0:
        print('%d iterations' % (i + 1))
        print('recon_error: %.3f' % np.mean(train_res_recon_error[-100:]))
        print('perplexity: %.3f' % np.mean(train_res_perplexity[-100:]))
        print()

