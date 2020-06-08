from binary_snn.utils_binary.training_utils import feedforward_sampling, local_feedback_and_update
from binary_snn.utils_binary.misc import refractory_period
from wispike.utils.misc import example_to_framed, framed_to_example
import torch
import numpy as np
import torch.nn.functional as F
from binary_snn.models.SNN import SNNetwork
from wispike.models.mlp import MLP
from binary_snn.utils_binary import misc as misc_snn
from wispike.models.vqvae import Model
import torch.optim as optim
from utils.filters import get_filter
from binary_snn.utils_binary.training_utils import init_training
import pyldpc


### VQ-VAE & LDPC
def init_vqvae(args):
    num_input_channels = args.dataset.root.stats.train_data[-1] // args.n_frames

    num_hiddens = 128
    num_residual_hiddens = 32
    num_residual_layers = 2

    commitment_cost = 0.25

    decay = 0.99

    learning_rate = 1e-3

    vqvae = Model(num_input_channels, num_hiddens, num_residual_layers, num_residual_hiddens, args.num_embeddings, args.embedding_dim, commitment_cost, decay).to(args.device)
    optimizer = optim.Adam(vqvae.parameters(), lr=learning_rate, amsgrad=False)

    return vqvae, optimizer


def train_vqvae(model, optimizer, args, train_res_recon_error, train_res_perplexity, example_idx):
    example = torch.FloatTensor(args.dataset.root.train.data[example_idx])
    framed = example_to_framed(example, args)

    optimizer.zero_grad()

    vq_loss, data_recon, perplexity = model(framed)
    recon_error = F.mse_loss(data_recon, framed)
    loss = recon_error + vq_loss
    loss.backward()

    optimizer.step()

    train_res_recon_error.append(recon_error.item())
    train_res_perplexity.append(perplexity.item())

    return train_res_recon_error, train_res_perplexity


def init_ldpc(encodings_dim):
    ldpc_codewords_length = 2 * np.prod(encodings_dim)
    d_v = 2
    d_c = 4

    # Make LDPC
    H, G = pyldpc.make_ldpc(ldpc_codewords_length, d_v, d_c, systematic=True, sparse=True)
    n, k = G.shape

    assert k >= np.prod(encodings_dim)

    return H, G, k


### Classifiers
def init_classifier(args):
    if args.classifier == 'snn':
        n_input_neurons = args.dataset.root.stats.train_data[1]
        n_output_neurons = args.dataset.root.stats.train_label[1]

        classifier = SNNetwork(**misc_snn.make_network_parameters(n_input_neurons,
                                                                  n_output_neurons,
                                                                  args.n_h,
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

        args.eligibility_trace_output, args.eligibility_trace_hidden, \
            args.learning_signal, args.baseline_num, args.baseline_den, args.S_prime = init_training(classifier, args)

    if args.classifier == 'mlp':
        n_input_neurons = np.prod(args.dataset.root.stats.train_data[1:])
        n_output_neurons = args.dataset.root.stats.train_label[1]

        classifier = MLP(n_input_neurons, args.n_h, n_output_neurons)
        args.mlp_optimizer = torch.optim.SGD(classifier.parameters(), lr=0.001)
        args.mlp_criterion = torch.nn.CrossEntropyLoss()

    return classifier, args


def train_snn(network, args, sample):
    network.set_mode('train')

    S_prime = sample.shape[-1]

    refractory_period(network)
    for s in range(S_prime):
        # Feedforward sampling
        log_proba, ls_tmp = feedforward_sampling(network, sample[:, s], args.gamma, args.r)
        # Local feedback and update
        args.eligibility_trace_hidden, args.eligibility_trace_output, args.lr, args.baseline_num, args.baseline_den \
            = local_feedback_and_update(network, ls_tmp, args.eligibility_trace_hidden, args.eligibility_trace_output,
                                        args.learning_signal, args.baseline_num, args.baseline_den, args.lr, args.beta, args.kappa)

        return network, args


def train_mlp(model, example, label, optimizer, criterion):
    print('training example shape ', example.shape)

    # clear the gradients of all optimized variables
    optimizer.zero_grad()

    # forward pass: compute predicted outputs by passing inputs to the model
    output = model(example)

    # calculate the loss
    loss = criterion(output.unsqueeze(0), label.unsqueeze(0))

    # backward pass: compute gradient of the loss with respect to model parameters
    loss.backward()
    # perform a single optimization step (parameter update)
    optimizer.step()

    return model


def train_classifier(model, args, idx):
    if isinstance(model, SNNetwork):
        example = torch.cat((torch.FloatTensor(args.dataset.root.train.data[idx]),
                            torch.FloatTensor(args.dataset.root.train.label[idx])), dim=0).to(model.device)

        model, args = train_snn(model, args, example)

    elif isinstance(model, MLP):
        example = torch.FloatTensor(args.dataset.root.train.data[idx]).flatten()
        label = torch.tensor(np.argmax(np.sum(args.dataset.root.train.label[:][idx], axis=(-1)), axis=-1))

        model = train_mlp(model, example, label, args.mlp_optimizer, args.mlp_criterion)

    return model, args


### WiSpike
def init_training_wispike(encoder, decoder, args):
    encoder.set_mode('train')
    decoder.set_mode('train')

    eligibility_trace_hidden_enc = {parameter: encoder.gradients[parameter]for parameter in encoder.gradients}
    eligibility_trace_hidden_dec = {parameter: decoder.gradients[parameter][decoder.hidden_neurons - decoder.n_input_neurons] for parameter in decoder.gradients}
    eligibility_trace_output_dec = {parameter: decoder.gradients[parameter][decoder.output_neurons - decoder.n_input_neurons] for parameter in decoder.gradients}

    learning_signal = 0

    baseline_num_enc = {parameter: eligibility_trace_hidden_enc[parameter].pow(2) * learning_signal for parameter in eligibility_trace_hidden_enc}
    baseline_den_enc = {parameter: eligibility_trace_hidden_enc[parameter].pow(2) for parameter in eligibility_trace_hidden_enc}

    baseline_num_dec = {parameter: eligibility_trace_hidden_dec[parameter].pow(2) * learning_signal for parameter in eligibility_trace_hidden_dec}
    baseline_den_dec = {parameter: eligibility_trace_hidden_dec[parameter].pow(2) for parameter in eligibility_trace_hidden_dec}

    S_prime = args.dataset.root.train.data.shape[-1]

    return eligibility_trace_hidden_enc, eligibility_trace_hidden_dec, eligibility_trace_output_dec, \
            learning_signal, baseline_num_enc, baseline_den_enc, baseline_num_dec, baseline_den_dec, S_prime

