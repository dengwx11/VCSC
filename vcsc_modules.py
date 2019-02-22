import collections
from typing import Iterable

import torch
from torch import nn as nn
from torch.distributions import Normal




class FCLayers(nn.Module):
    r"""A helper class to build fully-connected layers for a neural network.

    :param n_in: The dimensionality of the input
    :param n_out: The dimensionality of the output
    :param n_layers: The number of fully-connected hidden layers
    :param n_hidden: The number of nodes per hidden layer
    :param dropout_rate: Dropout rate to apply to each of the hidden layers
    """

    def __init__(self, n_in: int, n_out: int, 
                 n_layers: int = 1, n_hidden: int = 128, dropout_rate: float = 0.1, use_batch_norm=True):
        super(FCLayers, self).__init__()
        layers_dim = [n_in] + (n_layers - 1) * [n_hidden] + [n_out]

        self.fc_layers = nn.Sequential(collections.OrderedDict(
            [('Layer {}'.format(i), nn.Sequential(
                nn.Linear(n_in + sum(self.n_cat_list), n_out),
                nn.BatchNorm1d(n_out, momentum=.01, eps=0.001) if use_batch_norm else None,
                nn.ReLU(),
                nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None))
             for i, (n_in, n_out) in enumerate(zip(layers_dim[:-1], layers_dim[1:]))]))

    def forward(self, x: torch.Tensor):
        r"""Forward computation on ``x``.

        :param x: tensor of values with shape ``(n_in,)``
        :return: tensor of shape ``(n_out,)``
        :rtype: :py:class:`torch.Tensor`
        """
        
        for layers in self.fc_layers:
            for layer in layers:
                if layer is not None:
                    if isinstance(layer, nn.BatchNorm1d):
                        if x.dim() == 3:
                            x = torch.cat([(layer(slice_x)).unsqueeze(0) for slice_x in x], dim=0)
                        else:
                            x = layer(x)
                    else:
                        x = layer(x)
        return x


# Encoder
class Encoder(nn.Module):
    r"""Encodes data of ``n_input`` dimensions into a latent space of ``n_output``
    dimensions using a fully-connected neural network of ``n_hidden`` layers.

    :param n_input: The dimensionality of the input (data space)
    :param n_output: The dimensionality of the output (latent space)
    :param n_layers: The number of fully-connected hidden layers
    :param n_hidden: The number of nodes per hidden layer
    :dropout_rate: Dropout rate to apply to each of the hidden layers
    """

    def __init__(self, n_input: int, n_output: int,
                 n_layers: int = 1,
                 n_hidden: int = 128, dropout_rate: float = 0.1):
        super(Encoder, self).__init__()

        self.encoder = FCLayers(n_in=n_input, n_out=n_hidden, n_layers=n_layers,
                                n_hidden=n_hidden, dropout_rate=dropout_rate)
        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)

    def reparameterize(self, mu, var):
        return Normal(mu, var.sqrt()).rsample()

    def forward(self, x: torch.Tensor):
        r"""The forward computation for a single sample.

         #. Encodes the data into latent space using the encoder network
         #. Generates a mean \\( q_m \\) and variance \\( q_v \\) (clamped to \\( [-5, 5] \\))
         #. Samples a new value from an i.i.d. multivariate normal \\( \\sim N(q_m, \\mathbf{I}q_v) \\)

        :param x: tensor with shape (n_input,)
        :return: tensors of shape ``(n_latent,)`` for mean and var, and sample
        :rtype: 3-tuple of :py:class:`torch.Tensor`
        """

        # Parameters for latent distribution
        q = self.encoder(x, *cat_list)
        q_m = self.mean_encoder(q)
        q_v = torch.exp(self.var_encoder(q))  # (computational stability safeguard)torch.clamp(, -5, 5)
        latent = self.reparameterize(q_m, q_v)
        return q_m, q_v, latent


# Decoder
class DecoderVCSC(nn.Module):
    r"""Decodes data from latent space of ``n_input`` dimensions ``n_output``
    dimensions using a fully-connected neural network of ``n_hidden`` layers.

    :param n_input: The dimensionality of the input (latent space)
    :param n_output: The dimensionality of the output (data space). It is also the number of genesd
    :param n_cat_list: A list containing the number of categories
    :param n_layers: The number of fully-connected hidden layers
    :param n_hidden: The number of nodes per hidden layer
    :param dropout_rate: Dropout rate to apply to each of the hidden layers
    """

    def __init__(self, n_input: int, n_output: int,
                 n_layers: int = 1,
                 n_hidden: int = 128):
        super(DecoderVCSC, self).__init__()
        # dispersion: here we only deal with gene-cell dispersion case
        self.px_c_decoder = nn.Linear(n_input, n_output)

        # shape: shape parameter in gamma distribution infered from estimated covariance matrix
        self.px_sigma_decoder = FCLayers(n_in = n_output * (n_output - 1)/2, n_out = n_hidden, 
                                    n_layers=n_layers, n_hidden=n_hidden, dropout_rate=0)

        # scale: scale parameter in gamma distribution infered from estimated covariance matrix
        self.px_scale_decoder = nn.Sequential(nn.Linear(n_hidden, n_output), nn.Softmax(dim=-1))

        # shape: shape parameter in gamma distribution infered from estimated covariance matrix
        self.px_shape_decoder = nn.Linear(n_hidden, n_output)

        # dropout
        self.px_dropout_decoder = nn.Linear(n_hidden, n_output)


    def SDP(self, c: torch.Tensor, alpha: float = 1.0):
        px_sigma = []
        if c.dim() == 2:
            gene_num = c.shape[1]
        else:
            print("the dimension of c is not 2")
        for i in range(gene_num):
            for j in range(i+1,gene_num):
                px_sigma += [torch.exp(-(c[i]-c[j])*alpha).float()]
        px_sigma = torch.cat([element.unsqueeze(0) for element in px_sigma], dim=0)
        return px_sigma


    def forward(self, z: torch.Tensor):
        r"""The forward computation for a single sample.

         #. Decodes the data from the latent space using the decoder network
         #. Returns parameters for the ZINB distribution of expression

        :param z: tensor with shape ``(n_input,)``
        :param theta: empirical dispersion parameter in Gamma distribution with shape ``(#gene,)``
        :return: parameters for the ZINB distribution of expression
        :rtype: 4-tuple of :py:class:`torch.Tensor`
        """

        # scale parameter in gamma distribution: \theta
        

        # The decoder returns values for the parameters of the ZINB distribution
        px_c = self.px_c_decoder(z)
        px_sigma = self.SDP(px_c)
        px_dropout = self.px_dropout_decoder(px_sigma)
        px_shape = self.px_shape_decoder(px_sigma)
        px_scale = self.px_scale_decoder(px_sigma)
        
        return px_shape, px_scale, px_dropout, px_sigma


