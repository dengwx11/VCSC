# -*- coding: utf-8 -*-
"""Main module."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence as kl

#import log_zinb_positive, log_nb_positive
from vccs_modules import Encoder, DecoderVCSC


torch.backends.cudnn.benchmark = True


# VAE model
class VAE(nn.Module):
    r"""Variational auto-encoder model.

    :param n_input: Number of input genes
    :param n_batch: Number of batches
    :param n_labels: Number of labels
    :param n_hidden: Number of nodes per hidden layer
    :param n_latent: Dimensionality of the latent space
    :param n_layers: Number of hidden layers used for encoder and decoder NNs
    :param dropout_rate: Dropout rate for neural networks

    :param log_variational: Log variational distribution
    :param reconstruction_loss:  One of

        * ``'gamma'`` - gamma distribution
        * ``'zi-gamma'`` - Zero-inflated gamma distribution

    Examples:
        >>> gene_dataset = CortexDataset()
        >>> vae = VAE(gene_dataset.nb_genes, n_batch=gene_dataset.n_batches * False,
        ... n_labels=gene_dataset.n_labels)

    """

    def __init__(self, n_input: int, n_labels: int = 18,
                 n_hidden: int = 128, n_latent: int = 10, n_layers: int = 1,
                 dropout_rate: float = 0.1, 
                 log_variational: bool = True, reconstruction_loss: str = "zinb"):
        super(VAE, self).__init__()
        self.n_latent = n_latent
        self.log_variational = log_variational
        self.reconstruction_loss = reconstruction_loss
        # Automatically deactivate if useless
        self.n_labels = n_labels
        self.n_latent_layers = 1  # not sure what this is for, no usages?

        # if self.dispersion == "gene":
        #     self.px_r = torch.nn.Parameter(torch.randn(n_input, ))
        # elif self.dispersion == "gene-batch":
        #     self.px_r = torch.nn.Parameter(torch.randn(n_input, n_batch))
        # elif self.dispersion == "gene-label":
        #     self.px_r = torch.nn.Parameter(torch.randn(n_input, n_labels))
        # else:  # gene-cell
        #     pass

        # z encoder goes from the n_input-dimensional data to an n_latent-d
        # latent space representation
        self.z_encoder = Encoder(n_input, n_latent, n_layers=n_layers, n_hidden=n_hidden,
                                 dropout_rate=dropout_rate)
        # decoder goes from n_latent-dimensional space to n_input-d data
        self.decoder = DecoderVCSC(n_latent, n_input, n_cat_list=[n_batch], n_layers=n_layers, n_hidden=n_hidden)

    def get_latents(self, x):
        r""" returns the result of ``sample_from_posterior_z`` inside a list

        :param x: tensor of values with shape ``(batch_size, n_input)``
        :param y: tensor of cell-types labels with shape ``(batch_size, n_labels)``
        :return: one element list of tensor
        :rtype: list of :py:class:`torch.Tensor`
        """
        return [self.sample_from_posterior_z(x)]

    def sample_from_posterior_z(self, x, give_mean=False):
        r""" samples the tensor of latent values from the posterior
        #doesn't really sample, returns the means of the posterior distribution

        :param x: tensor of values with shape ``(batch_size, n_input)``
        :param y: tensor of cell-types labels with shape ``(batch_size, n_labels)``
        :param give_mean: is True when we want the mean of the posterior  distribution rather than sampling
        :return: tensor of shape ``(batch_size, n_latent)``
        :rtype: :py:class:`torch.Tensor`
        """
        if self.log_variational:
            pass
        else:
            x= exp(x)-1
        qz_m, qz_v, z = self.z_encoder(x)  
        if give_mean:
            z = qz_m
        return z

    def get_sample_shape(self, x, batch_index=None, y=None, n_samples=1):
        r"""Returns the tensor of predicted frequencies of expression

        :param x: tensor of values with shape ``(batch_size, n_input)``
        :param batch_index: array that indicates which batch the cells belong to with shape ``batch_size``
        :param y: tensor of cell-types labels with shape ``(batch_size, n_labels)``
        :param n_samples: number of samples
        :return: tensor of predicted frequencies of expression with shape ``(batch_size, n_input)``
        :rtype: :py:class:`torch.Tensor`
        """
        return self.inference(x)[0]

    def get_sample_scale(self, x, batch_index=None, y=None, n_samples=1):
        r"""Returns the tensor of means of the negative binomial distribution

        :param x: tensor of values with shape ``(batch_size, n_input)``
        :param y: tensor of cell-types labels with shape ``(batch_size, n_labels)``
        :param batch_index: array that indicates which batch the cells belong to with shape ``batch_size``
        :param n_samples: number of samples
        :return: tensor of means of the negative binomial distribution with shape ``(batch_size, n_input)``
        :rtype: :py:class:`torch.Tensor`
        """
        return self.inference(x)[1]

    def get_sample_sigma(self, x, batch_index=None, y=None, n_samples=1):
        r"""Returns the tensor of means of the negative binomial distribution

        :param x: tensor of values with shape ``(batch_size, n_input)``
        :param y: tensor of cell-types labels with shape ``(batch_size, n_labels)``
        :param batch_index: array that indicates which batch the cells belong to with shape ``batch_size``
        :param n_samples: number of samples
        :return: tensor of means of the negative binomial distribution with shape ``(batch_size, n_input)``
        :rtype: :py:class:`torch.Tensor`
        """
        return self.inference(x)[3]

    def _reconstruction_loss(self, x, px_shape, px_scale, px_dropout):
        # Reconstruction Loss
        if self.reconstruction_loss == 'zi-gamma':
            reconst_loss = -log_zigamma_positive(x, px_shape, px_scale, px_dropout)
        elif self.reconstruction_loss == 'gamma':
            reconst_loss = -log_gamma_positive(x, px_shape, px_scale)
        return reconst_loss

    def inference(self, x):
        x_ = x
        if self.log_variational:
            pass
        else:
            x_ = torch.exp(x_) - 1

        # Sampling
        qz_m, qz_v, z = self.z_encoder(x_)

        # if n_samples > 1:
        #     qz_m = qz_m.unsqueeze(0).expand((n_samples, qz_m.size(0), qz_m.size(1)))
        #     qz_v = qz_v.unsqueeze(0).expand((n_samples, qz_v.size(0), qz_v.size(1)))
        #     z = Normal(qz_m, qz_v.sqrt()).sample()
        #     ql_m = ql_m.unsqueeze(0).expand((n_samples, ql_m.size(0), ql_m.size(1)))
        #     ql_v = ql_v.unsqueeze(0).expand((n_samples, ql_v.size(0), ql_v.size(1)))
        #     library = Normal(ql_m, ql_v.sqrt()).sample()

        px_shape, px_scale, px_dropout, px_sigma = self.decoder(z)
        # if self.dispersion == "gene-label":
        #     px_r = F.linear(one_hot(y, self.n_labels), self.px_r)  # px_r gets transposed - last dimension is nb genes
        # elif self.dispersion == "gene-batch":
        #     px_r = F.linear(one_hot(batch_index, self.n_batch), self.px_r)
        # elif self.dispersion == "gene":
        #     px_r = self.px_r
        # px_r = torch.exp(px_r)

        return px_shape, px_scale, px_dropout, px_sigma, qz_m, qz_v, z

    def forward(self, x):
        r""" Returns the reconstruction loss and the Kullback divergences

        :param x: tensor of values with shape (batch_size, n_input)
        :param local_l_mean: tensor of means of the prior distribution of latent variable l
         with shape (batch_size, 1)
        :param local_l_var: tensor of variancess of the prior distribution of latent variable l
         with shape (batch_size, 1)
        :param batch_index: array that indicates which batch the cells belong to with shape ``batch_size``
        :param y: tensor of cell-types labels with shape (batch_size, n_labels)
        :return: the reconstruction loss and the Kullback divergences
        :rtype: 2-tuple of :py:class:`torch.FloatTensor`
        """
        # Parameters for z latent distribution

        px_shape, px_scale, px_dropout, px_sigma, qz_m, qz_v, z = self.inference(x)
        reconst_loss = self._reconstruction_loss(x, px_shape, px_scale, px_dropout)

        # KL Divergence
        mean = torch.zeros_like(qz_m)
        scale = torch.ones_like(qz_v)

        kl_divergence = kl(Normal(qz_m, torch.sqrt(qz_v)), Normal(mean, scale)).sum(dim=1)


        return reconst_loss  kl_divergence, px_sigma
