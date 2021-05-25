from __future__ import print_function
import tables
import yaml

from neurodata.load_data import create_dataloader
from snn.utils.misc import *
from snn.models.SNN import LayeredSNN
from snn.optimizer.snnsgd import SNNSGD
from snn.utils.utils_snn import refractory_period
from snn.utils.utils_snn import get_acc_layered

''''
'''

if __name__ == "__main__":
    # setting the hyper parameters
    parser = argparse.ArgumentParser(description='Train probabilistic multivalued SNNs using Pytorch')

    # Training arguments
    parser.add_argument('--home', default=r"C:\Users\K1804053\OneDrive - King's College London\PycharmProjects")
    parser.add_argument('--params_file', default='snn/snn/experiments/parameters/params_mnistdvs_binary_layered.yml')
    parser.add_argument('--save_path', type=str, default=None, help='Path to where weights are stored (relative to home)')
    parser.add_argument('--weights', type=str, default=None, help='Path to existing weights (relative to home)')

    # Arguments common to all models

    args = parser.parse_args()

print(args)

# Save results and weights
results_path = args.home + r'/results/'

if args.weights is None:
    if args.save_path is None:
        params_file = os.path.join(args.home, args.params_file)
        with open(params_file, 'r') as f:
            params = yaml.load(f)

        name = params['dataset'].split("/", -1)[-1][:-5] + r'_' + params['model'] \
               + r'_%d_epochs_dt_%d_' % (params['n_examples_train'], params['dt']) + r'_pol_' + str(params['polarity']) + params['suffix']

        args.save_path = mksavedir(pre=results_path, exp_dir=name)

        with open(args.save_path + '/params.yml', 'w') as outfile:
            yaml.dump(params_file, outfile, default_flow_style=False)
else:
    args.save_path = args.home + args.weights
    params_file = args.save_path + '/params.yml'

    with open(params_file, 'r') as f:
        params = yaml.load(f)


# Create dataloaders
dataset_path = args.home + params['dataset']
dataset = tables.open_file(dataset_path)
x_max = dataset.root.stats.train_data[1] // params['ds']
size = [(1 + params['polarity']) * x_max * x_max]

dataset.close()

### Network parameters
params['n_classes'] = len(params['classes'])
params['n_output_neurons'] = params['n_classes']

train_dl, test_dl = create_dataloader(dataset_path, batch_size=1, size=size, classes=params['classes'],
                                      sample_length_train=params['sample_length_train'], sample_length_test=params['sample_length_test'], dt=params['dt'],
                                      polarity=params['polarity'], ds=params['ds'], shuffle_test=True, num_workers=0)
T = int(params['sample_length_train'] / params['dt'])

# Prepare placeholders for recording test/train acc/loss
_, _, test_accs, _ = make_recordings(args, params)

if not params['disable_cuda'] and torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')

print(args.device)

network = LayeredSNN(size[0], params['n_neurons_per_layer'], params['n_classes'],
                     synaptic_filter=filters.raised_cosine_pillow_08, n_basis_feedforward=[8],
                     n_basis_feedback=[1], tau_ff=[10], tau_fb=[10], mu=[0.5], device=args.device).to(args.device)

print([w.shape for w in network.parameters()])
print(network.out_layer.ff_weights.device)

optimizer = SNNSGD([{'params': network.out_layer.parameters(), 'ls': False, 'baseline': False},
                    {'params': network.hidden_layers.parameters(), 'ls': True, 'baseline': True}
                    ], lr=params['lr'])

loss_fn = torch.nn.BCELoss(reduction='mean')

network.train()

train_iterator = iter(train_dl)

acc_best = 0.

for trial in range(params['num_trials']):
    for ite in range(params['n_examples_train']):
        if (ite+1) % params['test_period'] == 0:
            print('Ite %d: ' % (ite+1))
            acc_layered = get_acc_layered(network, test_dl, len(iter(test_dl)), T)
            print('Acc: %d', acc_layered)

            test_accs[int(ite + 1)].append(acc_layered)

            with open(args.save_path + '/test_accs.pkl', 'wb') as f:
                pickle.dump(test_accs, f, pickle.HIGHEST_PROTOCOL)

        network.train(args.save_path)

        refractory_period(network)

        try:
            inputs, targets = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_dl)
            inputs, targets = next(train_iterator)

        inputs = inputs[0].to(network.device)
        targets = targets[0].to(network.device)

        for t in range(T):
            ### LayeredSNN
            net_probas, net_outputs, probas_hidden, outputs_hidden = network(inputs[:t].T, targets[:, t], n_samples=params['n_samples'])

            # Generate gradients and KL regularization for hidden neurons
            out_loss = loss_fn(net_probas, net_outputs)
            if probas_hidden is not None:
                hidden_loss = loss_fn(probas_hidden, outputs_hidden.detach())
                with torch.no_grad():
                    kl_reg = params['gamma'] * torch.mean(outputs_hidden * torch.log(1e-7 + probas_hidden / params['r'])
                                                          + (1 - outputs_hidden) * torch.log(1e-7 + (1 - probas_hidden) / (1 - params['r'])))
            else:
                hidden_loss = 0
                kl_reg = 0

            loss = out_loss + hidden_loss
            loss.backward()

            optimizer.step(out_loss.detach() + kl_reg)
            optimizer.zero_grad()
