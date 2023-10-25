#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Variational Dropout version of linear and convolutional layers


Karen Ullrich, Christos Louizos, Oct 2017
"""

import math

import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch import nn
from torch.nn.modules import Module
from torch.autograd import Variable
from torch.nn.modules import utils


def reparametrize(mu, logvar, sampling=True):
    if sampling:
        std = logvar.mul(0.5).exp_()
        eps = torch.randn(*std.size(), device=mu.device)
        return mu + eps * std
    else:
        return mu


# -------------------------------------------------------
# LINEAR LAYER
# -------------------------------------------------------

class LinearGroupNJ(Module):
    """Fully Connected Group Normal-Jeffrey's layer (aka Group Variational Dropout).

    References:
    [1] Kingma, Diederik P., Tim Salimans, and Max Welling. "Variational dropout and the local reparameterization trick." NIPS (2015).
    [2] Molchanov, Dmitry, Arsenii Ashukha, and Dmitry Vetrov. "Variational Dropout Sparsifies Deep Neural Networks." ICML (2017).
    [3] Louizos, Christos, Karen Ullrich, and Max Welling. "Bayesian Compression for Deep Learning." NIPS (2017).
    """

    def __init__(self, in_features, out_features, cuda=False, init_weight=None, init_bias=None, clip_var=None):

        super(LinearGroupNJ, self).__init__()
        self.cuda = cuda
        self.in_features = in_features
        self.out_features = out_features
        self.clip_var = clip_var
        self.deterministic = False  # flag is used for compressed inference
        # trainable params according to Eq.(6)
        # dropout params
        self.z_mu = Parameter(torch.Tensor(in_features))
        self.z_logvar = Parameter(torch.Tensor(in_features))  # = z_mu^2 * alpha
        # weight params
        self.weight_mu = Parameter(torch.Tensor(out_features, in_features))
        self.weight_logvar = Parameter(torch.Tensor(out_features, in_features))

        self.bias_mu = Parameter(torch.Tensor(out_features))
        self.bias_logvar = Parameter(torch.Tensor(out_features))

        # init params either random or with pretrained net
        self.reset_parameters(init_weight, init_bias)

        # activations for kl
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()

        # numerical stability param
        self.epsilon = 1e-8

    def reset_parameters(self, init_weight, init_bias):
        # init means
        stdv = 1. / math.sqrt(self.weight_mu.size(1))

        self.z_mu.data.normal_(1, 1e-2)

        if init_weight is not None:
            self.weight_mu.data = torch.Tensor(init_weight)
        else:
            self.weight_mu.data.normal_(0, stdv)

        if init_bias is not None:
            self.bias_mu.data = torch.Tensor(init_bias)
        else:
            self.bias_mu.data.fill_(0)

        # init logvars
        self.z_logvar.data.normal_(-9, 1e-2)
        self.weight_logvar.data.normal_(-9, 1e-2)
        self.bias_logvar.data.normal_(-9, 1e-2)

    def clip_variances(self):
        if self.clip_var:
            self.weight_logvar.data.clamp_(max=math.log(self.clip_var))
            self.bias_logvar.data.clamp_(max=math.log(self.clip_var))

    def get_log_dropout_rates(self):
        log_alpha = self.z_logvar - torch.log(self.z_mu.pow(2) + self.epsilon)
        return log_alpha

    def compute_posterior_params(self):
        weight_var, z_var = self.weight_logvar.exp(), self.z_logvar.exp()
        self.post_weight_var = self.z_mu.pow(2) * weight_var + z_var * self.weight_mu.pow(2) + z_var * weight_var
        self.post_weight_mu = self.weight_mu * self.z_mu
        return self.post_weight_mu, self.post_weight_var

    def forward(self, x):
        if self.deterministic:
            assert self.training == False, "Flag deterministic is True. This should not be used in training."
            return F.linear(x, self.post_weight_mu, self.bias_mu)

        batch_size = x.size()[0]
        # compute z
        # note that we reparametrise according to [2] Eq. (11) (not [1])
        z = reparametrize(self.z_mu.repeat(batch_size, 1), self.z_logvar.repeat(batch_size, 1), sampling=self.training)

        # apply local reparametrisation trick see [1] Eq. (6)
        # to the parametrisation given in [3] Eq. (6)
        xz = x * z
        mu_activations = F.linear(xz, self.weight_mu, self.bias_mu)
        var_activations = F.linear(xz.pow(2), self.weight_logvar.exp(), self.bias_logvar.exp())

        return reparametrize(mu_activations, var_activations.log(), sampling=self.training)

    def kl_divergence(self):
        # KL(q(z)||p(z))
        # we use the kl divergence approximation given by [2] Eq.(14)
        k1, k2, k3 = 0.63576, 1.87320, 1.48695
        log_alpha = self.get_log_dropout_rates()
        KLD = -torch.sum(k1 * self.sigmoid(k2 + k3 * log_alpha) - 0.5 * self.softplus(-log_alpha) - k1)

        # KL(q(w|z)||p(w|z))
        # we use the kl divergence given by [3] Eq.(8)
        KLD_element = -0.5 * self.weight_logvar + 0.5 * (self.weight_logvar.exp() + self.weight_mu.pow(2)) - 0.5
        KLD += torch.sum(KLD_element)

        # KL bias
        KLD_element = -0.5 * self.bias_logvar + 0.5 * (self.bias_logvar.exp() + self.bias_mu.pow(2)) - 0.5
        KLD += torch.sum(KLD_element)

        return KLD

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
