from binary_snn.utils_binary.training_utils import feedforward_sampling, local_feedback_and_update
from binary_snn.utils_binary.misc import refractory_period
import torch
from binary_snn.models.SNN import SNNetwork
from wispike.models.mlp import MLP
import numpy as np
import torch.nn.functional as F


def train_snn(network, args, sample):
    network.set_mode('train')

    S_prime = sample.shape[-1]

    refractory_period(network)
    for s in range(S_prime):
        # Feedforward sampling
        log_proba, ls_tmp = feedforward_sampling(network, sample[:, s], args.alpha, args.r)
        # Local feedback and update
        args.eligibility_trace_hidden, args.eligibility_trace_output, args.learning_signal, args.baseline_num, args.baseline_den \
            = local_feedback_and_update(network, ls_tmp, args.eligibility_trace_hidden, args.eligibility_trace_output,
                                        args.learning_signal, args.baseline_num, args.baseline_den, args.learning_rate, args.beta, args.kappa)

        return args


def train_mlp():
    return


def train_classifier(model, optimizer, args, sample_idx, weights=None):
    if isinstance(model, SNNetwork):
        args.eligibility_trace_hidden = {parameter: model.gradients[parameter][model.hidden_neurons - model.n_input_neurons] for parameter in model.gradients}
        args.eligibility_trace_output = {parameter: model.gradients[parameter][model.output_neurons - model.n_input_neurons] for parameter in model.gradients}
        args.learning_signal = 0

        args.baseline_num = {parameter: args.eligibility_trace_hidden[parameter].pow(2) * args.learning_signal for parameter in args.eligibility_trace_hidden}
        args.baseline_den = {parameter: args.eligibility_trace_hidden[parameter].pow(2) for parameter in args.eligibility_trace_hidden}

        sample = torch.cat((torch.FloatTensor(args.dataset.root.train.data[sample_idx]),
                            torch.FloatTensor(args.dataset.root.train.label[sample_idx])), dim=0).to(model.device)

        model, args = train_snn(model, args, sample)

    elif isinstance(model, MLP):
        sample = torch.FloatTensor(args.dataset.root.train.data[sample_idx]).flatten()
        label = torch.tensor(np.argmax(np.sum(args.dataset.root.test.label[:], axis=(-1)), axis=-1))

        model = train_mlp(model, optimizer, args, sample, label)

    return model, args


def train_vqvae(model, optimizer, args, train_res_recon_error, train_res_perplexity,  sample_idx):
    if args.residual:
        data = torch.zeros([args.n_frames, 80 // (args.n_frames - 1), 26, 26])
        data[:-1] = torch.FloatTensor(args.dataset.root.train.data[sample_idx, :, :(args.n_frames - 1) * (80 // (args.n_frames - 1))]).transpose(1, 0).unsqueeze(0).unsqueeze(3).reshape(data[:-1].shape)
        data[-1, :args.residual] = torch.FloatTensor(args.dataset.root.train.data[sample_idx, :, :-args.residual]).transpose(1, 0).unsqueeze(0).unsqueeze(3).reshape([1, args.residual, 26, 26])
    else:
        data = torch.FloatTensor(args.dataset.root.train.data[sample_idx, :, :]).transpose(1, 0).unsqueeze(0).unsqueeze(3).reshape([args.n_frames, 80 // args.n_frames, 26, 26])

    optimizer.zero_grad()

    vq_loss, data_recon, perplexity = model(data)
    recon_error = F.mse_loss(data_recon, data)
    loss = recon_error + vq_loss
    loss.backward()

    optimizer.step()

    train_res_recon_error.append(recon_error.item())
    train_res_perplexity.append(perplexity.item())

