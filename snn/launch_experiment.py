from __future__ import print_function
import tables
import yaml

from neurodata.load_data import create_dataloader
from snn.utils.misc import *
from snn.experiments import binary_exp, wta_exp

''''
'''

if __name__ == "__main__":
    # setting the hyper parameters
    parser = argparse.ArgumentParser(description='Train probabilistic multivalued SNNs using Pytorch')

    # Training arguments
    parser.add_argument('--home', default=r"\home")
    parser.add_argument('--params_file', default='snn\snn\experiments\parameters\params_mnistdvs_binary.yml')
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
               + r'_%d_epochs_nh_%d_dt_%d_' % (params['n_examples_train'], params['n_hidden_neurons'], params['dt']) + r'_pol_' + str(params['polarity']) + params['suffix']

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

dataset.close()

### Network parameters
params['n_classes'] = len(params['classes'])
params['n_output_neurons'] = params['n_classes']

if params['model'] == 'snn':
    params['n_input_neurons'] = (1 + params['polarity']) * x_max * x_max
    size = [params['n_input_neurons']]
elif params['model'] == 'wta':
    params['n_input_neurons'] = x_max * x_max
    size = [2, params['n_input_neurons']]
else:
    raise NotImplementedError

train_dl, test_dl = create_dataloader(dataset_path, batch_size=1, size=size, classes=params['classes'],
                                      sample_length_train=params['sample_length_train'], sample_length_test=params['sample_length_test'], dt=params['dt'],
                                      polarity=params['polarity'], ds=params['ds'], shuffle_test=True, num_workers=0)

# Prepare placeholders for recording test/train acc/loss
train_accs, train_losses, test_accs, test_losses = make_recordings(args, params)

args.device = None
if not params['disable_cuda'] and torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')




if params['topology_type'] == 'custom':
    params['topology'] = torch.zeros([args.n_hidden_neurons + args.n_output_neurons,
                                 args.n_input_neurons + args.n_hidden_neurons + args.n_output_neurons])
    params['topology'][-args.n_output_neurons:, args.n_input_neurons:-args.n_output_neurons] = 1
    params['topology'][:args.n_hidden_neurons, :(args.n_input_neurons + args.n_hidden_neurons)] = 1
    # Feel free to fill this with any custom topology
    print(params['topology'])
else:
    params['topology'] = None

# Create the network
if params['model'] == 'snn':
    binary_exp.launch_binary_exp(args, params, train_dl, test_dl, train_accs, train_losses, test_accs, test_losses)
elif params['model'] == 'wta':
    wta_exp.launch_multivalued_exp(args, params, train_dl, test_dl, train_accs, train_losses, test_accs, test_losses)
else:
    raise NotImplementedError('Please choose a model between "snn" and "wta"')

