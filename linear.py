import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import  Normal, Laplace
from torch.distributions.kl import kl_divergence
from torch.nn import init, Parameter

from torch.nn.modules import Module
from torch.autograd import Variable
from torch.nn.modules import utils

class LinearGroupHS(nn.Module):
    def __init__(self, in_features, out_features, tau_0=1e-5):
        super(LinearGroupHS, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.tau_0 = tau_0

        self.sa_mu = nn.Parameter(torch.Tensor(1), requires_grad=True)
        self.sa_logvar = nn.Parameter(torch.Tensor(1), requires_grad=True)

        self.sb_mu = nn.Parameter(torch.Tensor(1), requires_grad=True)
        self.sb_logvar = nn.Parameter(torch.Tensor(1), requires_grad=True)

        self.alpha_mu = nn.Parameter(torch.Tensor(in_features), requires_grad=True)
        self.alpha_logvar = nn.Parameter(torch.Tensor(in_features), requires_grad=True)

        self.beta_mu = nn.Parameter(torch.Tensor(in_features), requires_grad=True)
        self.beta_logvar = nn.Parameter(torch.Tensor(in_features), requires_grad=True)

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features), requires_grad=True)
        self.weight_logvar = nn.Parameter(torch.Tensor(out_features, in_features), requires_grad=True)

        self.bias_mu = nn.Parameter(torch.Tensor(out_features), requires_grad=True)
        self.bias_logvar = nn.Parameter(torch.Tensor(out_features), requires_grad=True)

        self.reset_parameters()

        self.epsilon = 1e-8

    def reset_parameters(self):
        self.sa_mu.data.normal_(-3, 1e-2)
        self.sb_mu.data.normal_(0, 1e-2)
        self.alpha_mu.data.normal_(0, 1e-2)
        self.beta_mu.data.normal_(0, 1e-2)

        self.weight_mu.data.normal_(0, 1e-2)
        torch.nn.init.orthogonal_(self.weight_mu)
        self.bias_mu.data.fill_(0)

        # init logvars
        self.sa_logvar.data.normal_(-9, 1e-2)
        self.sb_logvar.data.normal_(0, 1e-2)
        self.alpha_logvar.data.normal_(-9, 1e-2)
        self.beta_logvar.data.normal_(0, 1e-2)
        self.weight_logvar.data.normal_(-9, 1e-2)
        self.bias_logvar.data.normal_(-9, 1e-2)


    def forward(self, x, n_samples=100):
        batch_size = x.size()[-2]

        s_mu = 0.5 * self.sa_mu + 0.5 * self.sb_mu
        s_scale = torch.sqrt(0.25*self.sa_logvar.exp() + 0.25*self.sb_logvar.exp())

        if self.training:
            eps = x.new_zeros(batch_size, self.in_features).normal_()
            log_s = s_mu + s_scale * eps
        else:
            eps = x.new_zeros(n_samples, batch_size, self.in_features).normal_()
            log_s = s_mu + s_scale * eps

        z_mu = 0.5*self.alpha_mu.repeat(batch_size, 1) + 0.5*self.beta_mu.repeat(batch_size, 1) + log_s
        z_scale = torch.sqrt(0.25*self.alpha_logvar.exp().repeat(batch_size, 1) + 0.25*self.beta_logvar.exp().repeat(batch_size, 1))

        if self.training:
            eps = x.new_zeros(batch_size, self.in_features).normal_()
            z = torch.exp(z_mu + z_scale * eps)
        else:
            eps = x.new_zeros(n_samples, batch_size, self.in_features).normal_()
            z = torch.exp(z_mu + z_scale * eps)

        xz = x * z
        mu_activations = F.linear(xz, self.weight_mu, self.bias_mu)
        var_activations = F.linear(xz.pow(2), self.weight_logvar.exp(), self.bias_logvar.exp())

        if self.training:
            eps = x.new_zeros(batch_size, self.out_features).normal_()
        else:
            eps = x.new_zeros(n_samples, batch_size, self.out_features).normal_()

        return (mu_activations + torch.sqrt(var_activations) * eps)

    def kl_divergence(self):
        # KL(q(z)||p(z))
        KLD = -math.log(self.tau_0) + (torch.exp(self.sa_mu + 0.5 * self.sa_logvar.exp()) / self.tau_0) - 0.5 * (self.sa_mu + self.sa_logvar + 1 + math.log(2.0))
        KLD += torch.exp(0.5 * self.sb_logvar.exp() - self.sb_mu) - 0.5 * (-self.sb_mu + self.sb_logvar + 1 + math.log(2.0))

        KLD_element = -1.0 * (-torch.exp(self.alpha_mu + 0.5 * self.alpha_logvar.exp()) + 0.5 * (self.alpha_mu + self.alpha_logvar + 1 + math.log(2.0)))
        KLD += torch.sum(KLD_element)

        KLD_element = -1.0 * (-torch.exp(0.5 * self.beta_logvar.exp() - self.beta_mu) + 0.5 * (-self.beta_mu + self.beta_logvar + 1 + math.log(2.0)))
        KLD += torch.sum(KLD_element)

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

class LinearGaussian(nn.Module):
    def __init__(self, in_features, out_features, std=0.1):
        super(LinearGaussian, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.prior = Normal(torch.zeros(1).cuda(), std * torch.ones(1).cuda())

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features), requires_grad=True)
        self.weight_logvar = nn.Parameter(torch.Tensor(out_features, in_features), requires_grad=True)

        self.bias_mu = nn.Parameter(torch.Tensor(out_features), requires_grad=True)
        self.bias_logvar = nn.Parameter(torch.Tensor(out_features), requires_grad=True)

        self.reset_parameters()

        self.epsilon = 1e-8

    def reset_parameters(self):
        torch.nn.init.orthogonal_(self.weight_mu)
        self.bias_mu.data.normal_(0, 1e-2)

        # init logvars
        self.weight_logvar.data.normal_(-9, 1e-2)
        self.bias_logvar.data.normal_(-9, 1e-2)


    def forward(self, x, n_samples=100):
        batch_size = x.size()[-2]

        mu_activations = F.linear(x, self.weight_mu, self.bias_mu)
        var_activations = F.linear(x.pow(2), self.weight_logvar.exp(), self.bias_logvar.exp())

        if self.training:
            eps = x.new_zeros(batch_size, self.out_features).normal_()
        else:
            eps = x.new_zeros(n_samples, batch_size, self.out_features).normal_()

        return (mu_activations + torch.sqrt(var_activations) * eps)

    def kl_divergence(self):

        weight_dist = Normal(self.weight_mu, self.weight_logvar.exp().sqrt())
        bias_dist = Normal(self.bias_mu, self.bias_logvar.exp().sqrt())

        return kl_divergence(weight_dist, self.prior).sum() + kl_divergence(bias_dist, self.prior).sum()



    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
