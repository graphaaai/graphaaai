import logging

import torch.nn.functional

from torch.nn import Dropout, Linear
from torch.nn.modules import Module

LOG = logging.getLogger(__name__)


class FCNetwork(Module):
    """
    A simple implementation of a multilayer fully-connected feed-forward neural network.
    Args:
        in_dim: The number of input dimensions.
        out_dim: The number of output dimensions.
        hidden_dims: List[int] of hidden dimensions of the layers, ordered from input to output. The length of the list
                     determines the number of hidden layers.
        activation: The name of the activation function. Should be one of {sigmoid, relu, tanh, elu}. If none, no
                    activation function is used (linear network).
        dropout: The fraction with which nodes should be dropped. If the value is 0, dropout is not applied.
                 Dropout is applied to all intermediate layers except the final layer projecting into `out_dim`.
        activate_last_layer: Whether the activation function should be applied to the last layer.
    """
    def __init__(self, in_dim, out_dim, hidden_dims, activation=None, dropout=0.4, activate_last_layer=False):
        super(FCNetwork, self).__init__()

        assert len(hidden_dims) > 0
        assert activation is None or activation in {'sigmoid', 'relu', 'tanh', 'elu'}

        self.activate_last_layer = activate_last_layer
        self.num_layers = len(hidden_dims) + 1
        self.dropout = dropout
        self.last_layer_index = len(hidden_dims)
        for layer_num in range(self.num_layers):
            layer = FCNetwork._get_layer(layer_num, in_dim, out_dim, hidden_dims)
            setattr(self, f'fc_layer_{layer_num}', layer)
            if dropout > 0:
                setattr(self, f'dropout_layer_{layer_num}', Dropout(dropout))

        self.activation = FCNetwork._get_activation(activation)

    @staticmethod
    def _get_activation(fn_name):
        if fn_name:
            if fn_name == 'elu':
                return torch.nn.functional.elu
            return getattr(torch, fn_name)

    @staticmethod
    def _get_layer(layer_no, in_dim, out_dim, hidden_dims):
        """
        Helper method to get the current layer with the appropriate number of dimensions.
        :return: A Linear layer of PyTorch.
        """
        if layer_no == 0:
            return Linear(in_dim, hidden_dims[0])
        elif layer_no == len(hidden_dims):
            return Linear(hidden_dims[-1], out_dim)
        else:
            return Linear(hidden_dims[layer_no - 1], hidden_dims[layer_no])

    def forward(self, x):
        for layer_num in range(self.num_layers):
            if self.dropout > 0:
                x = getattr(self, f'dropout_layer_{layer_num}')(x)
            x = getattr(self, f'fc_layer_{layer_num}')(x)
            # Don't activate the output of the last layer.
            if self.activation and (self.activate_last_layer or layer_num < self.last_layer_index):
                x = self.activation(x)

        return x
