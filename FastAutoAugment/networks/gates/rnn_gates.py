import torch
import torch.nn as nn
import math
from torch.autograd import Variable
import torch.autograd as autograd
from torch.distributions import Categorical


def repackage_hidden(h):
    """ to reduce memory usage"""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


class RNNGate(nn.Module):
    """Recurrent Gate definition.
    Input is already passed through average pooling and embedding."""
    def __init__(self, input_dim, hidden_dim, rnn_type='lstm'):
        super(RNNGate, self).__init__()
        self.rnn_type = rnn_type
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_dim, hidden_dim)
        else:
            self.rnn = None
        self.hidden = None

        # reduce dim
        self.proj = nn.Linear(hidden_dim, 1)
        self.proj.not_interesting = True
        self.prob = nn.Sigmoid()

    def init_hidden(self, batch_size):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim).cuda()),
                autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim).cuda()))

    def repackage_hidden(self):
        self.hidden = repackage_hidden(self.hidden)

    def forward(self, x):
        # Take the convolution output of each step
        batch_size = x.size(0)
        self.rnn.flatten_parameters()
        out, self.hidden = self.rnn(x.view(1, batch_size, -1), self.hidden)

        proj = self.proj(out.squeeze())
        prob = self.prob(proj)

        disc_prob = (prob > 0.5).float().detach() - \
                    prob.detach() + prob

        disc_prob = disc_prob.view(batch_size, 1, 1, 1)
        return disc_prob, prob


class RNNGatePolicy(nn.Module):
    def __init__(self, input_dim, hidden_dim, rnn_type='lstm'):
        super(RNNGatePolicy, self).__init__()

        self.rnn_type = rnn_type
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_dim, hidden_dim)
        else:
            self.rnn = None
        self.hidden = None

        # reduce dim. use softmax here for two actions.
        self.proj = nn.Linear(hidden_dim, 1)
        self.proj.not_interesting = True
        self.prob = nn.Sigmoid()

        # saved actions and rewards
        self.saved_actions = []
        self.rewards = []

    def hotter(self, t):
        self.proj.weight.data /= t
        self.proj.bias.data /= t

    def init_hidden(self, batch_size):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (autograd.Variable(torch.zeros(1, batch_size,
                                              self.hidden_dim).cuda()),
                autograd.Variable(torch.zeros(1, batch_size,
                                              self.hidden_dim).cuda()))

    def repackage_hidden(self):
        self.hidden = repackage_hidden(self.hidden)

    def forward(self, x):
        batch_size = x.size(0)
        self.rnn.flatten_parameters()
        out, self.hidden = self.rnn(x.view(1, batch_size, -1), self.hidden)
        proj = self.proj(out.squeeze())
        prob = self.prob(proj)
        bi_prob = torch.cat([1 - prob, prob], dim=1)

        if self.training:
            dist = Categorical(bi_prob)
            action = dist.sample()
            self.saved_actions.append(action)
        else:
            action = (prob > 0.5).float()
            self.saved_actions.append(action)
        action = action.view(action.size(0), 1, 1, 1).float()
        return action, bi_prob
