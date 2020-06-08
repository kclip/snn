import torch
import torch.nn.functional as F


class MLP(torch.nn.Module):
    def __init__(self, input_size, n_hidden, n_classes):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, n_hidden)
        self.fc2 = torch.nn.Linear(n_hidden, n_classes)

    def forward(self, x):
        # add hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        # add output layer
        x = self.fc2(x)
        return x
