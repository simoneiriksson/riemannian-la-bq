import torch
from torch import nn
from utils import extract_parameters, set_weights_old
from pyro import distributions as dist
import numpy as np
import pyro

class MyModel(torch.nn.Module):
    def __init__(self, M, N, fn):
        super(MyModel, self).__init__()
        self.M = M
        self.N = N
        self.weights = torch.nn.Parameter(torch.rand((M+1,1)))
        self.fn = fn
        
    def forward(self, xs):
        return fn(xs) @ self.weights

class mini(nn.Module):
    # Implemented based on https://github.com/ChawDoe/LeNet5-MNIST-PyTorch
    def __init__(self, seed=None, M_inputs=2, num_hid=10, num_out=10):
        super(mini, self).__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.fc1 = nn.Linear(M_inputs, num_hid, bias=False)
        self.fc2 = nn.Linear(num_hid, num_out, bias=False)
        self.nonlin = nn.ReLU()
    def forward(self, x):
        x = nn.Flatten(1)(x)
        x = self.fc1(x)
        x = self.nonlin(x)
        x = self.fc2(x)
        return x
    


class PyroModel(torch.nn.Module):

    def __init__(self, base, prior_log_sigma, likelihood_given_outputs, batch_size = 100, device="cpu", *args, **kwargs):

        super(PyroModel, self).__init__()

        self.base_model = base
        self.base_params = extract_parameters(self.base_model)

        #self.rank = cov_factor.size()[0]
        self.prior_log_sigma = prior_log_sigma
        
        self.likelihood = likelihood_given_outputs
        self.batch_size = batch_size
        self.model_size = sum([p[2].numel() for p in self.base_params])
        self.device = device

    def model(self, x, y):
        self.parameters_samples = pyro.sample("parameters_samples", dist.Normal(torch.zeros(self.model_size), self.model_size * np.exp(self.prior_log_sigma)).to_event(1)).to(self.device)
        bs = self.batch_size
        num_batches = x.shape[0] // bs
        if x.shape[0] % bs: num_batches += 1

        for i in pyro.plate("batches", num_batches): 
            x_ = x[i * bs: (i+1)*bs] 
            y_ = y[i * bs: (i+1)*bs] 
            with pyro.plate("data"+str(i), x_.shape[0]):
                set_weights_old(self.base_params, self.parameters_samples, self.device)
                z = self.base_model(x_)
                pyro.sample("y"+str(i), self.likelihood(z).to_event(1), obs=y_)

    def forward(self, *args, **kwargs):
        self.parameters_samples = pyro.sample("parameters_samples", dist.Normal(torch.zeros(self.model_size), self.model_size * np.exp(self.prior_log_sigma)).to_event(1)).to(self.device)
        set_weights_old(self.base_params, self.parameters_samples, self.device)
        return self.base_model(*args, **kwargs)