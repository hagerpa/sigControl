import torch
from torch import nn


class DNNStrategy(torch.nn.Module):
    def __init__(self, in_dim, nn_hidden=2, nn_dropout=0.0, **kwargs):
        """
        ReLU deep neural network
        :param in_dim: input dimension
        :param n_hidden: number of hidden layers
        """
        super(DNNStrategy, self).__init__()
        # _, d = x.shape
        hidden = in_dim + 30
        self.n_hidden = nn_hidden
        a_dim = in_dim
        layers = [nn.Dropout(nn_dropout)]

        for i in range(nn_hidden):
            dens_ = nn.Linear(a_dim, hidden, dtype=torch.float64)
            nn.init.xavier_uniform_(dens_.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.zeros_(dens_.bias)
            layers += [dens_, nn.ReLU()]
            a_dim = hidden

        final = torch.nn.Linear(a_dim, 1, dtype=torch.float64)
        nn.init.xavier_normal_(final.weight, gain=nn.init.calculate_gain('linear'))
        nn.init.zeros_(final.bias)
        layers += [final, torch.nn.Flatten()]

        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ConstraintLayer(torch.nn.Module):
    """
    Model layer that enforces a certain interval constrained.
    """
    def __init__(self, constraint=(-1, 1)):
        super(ConstraintLayer, self).__init__()
        self.constraint = constraint

    def forward(self, x):
        return torch.clamp(x, *self.constraint)


class TimeWeightedAverage(torch.nn.Module):
    """
    Baseline strategy in the optimal execution example.
    """
    def __init__(self, y0, T, kappa, kappa_T):
        super(TimeWeightedAverage, self).__init__()
        self.y0 = y0
        self.T = T
        self.kappa = kappa
        self.kappa_T = kappa_T
        self.dummy_param = torch.tensor(1.0, requires_grad=True)

    def forward(self, x):
        return torch.ones_like(x[:, 0], requires_grad=True) * self.y0 * self.kappa_T / (
                self.kappa + self.T * self.kappa_T)

    def parameters(self, recurse: bool = True):
        return [self.dummy_param]
