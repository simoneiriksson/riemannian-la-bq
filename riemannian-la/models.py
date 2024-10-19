import torch
from torch import nn
from utils import extract_parameters, set_weights_old
from pyro import distributions as dist
import numpy as np
import pyro

class MyModel(torch.nn.Module):
    def __init__(self, input_dim, num_classes, fn, prior_log_sigma):
        super(MyModel, self).__init__()
        self.input_dim = input_dim
        dummy = torch.zeros((1, input_dim))
        self.num_params = fn(dummy).shape[1]
        self.num_classes = num_classes
        self.weights = torch.nn.Parameter(torch.rand((self.num_params, num_classes))*prior_log_sigma)
        #self.weights = torch.nn.Parameter(torch.rand((3,1))*prior_log_sigma)
        self.fn = fn

    def forward(self, xs):
        return self.fn(xs) @ self.weights

class mini(nn.Module):
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
    

# Adapted from https://github.com/wjmaddox/drbayes/blob/master/subspace_inference/posteriors/pyro.py
class PyroModel(torch.nn.Module):

    def __init__(self, base, prior_log_sigma, likelihood_given_outputs, batch_size = None, device="cpu", *args, **kwargs):

        super(PyroModel, self).__init__()

        self.base_model = base
        self.base_params = extract_parameters(self.base_model)

        #self.rank = cov_factor.size()[0]
        self.prior_log_sigma = prior_log_sigma
        
        self.likelihood = likelihood_given_outputs
        self.batch_size = batch_size
        self.model_size = sum([p[2].numel() for p in self.base_params])
        self.device = device

    def model(self, x, y=None):
        self.parameters_samples = pyro.sample("parameters_samples", 
                                              dist.Normal(torch.zeros(self.model_size), 
                                                          torch.ones(self.model_size) * np.exp(self.prior_log_sigma)).to_event(1)).to(self.device)
        bs = self.batch_size
        if bs is not None: 
            num_batches = x.shape[0] // bs
            if x.shape[0] % bs: num_batches += 1
            
            for i in pyro.plate("batches", num_batches): 
                x_ = x[i * bs: (i+1)*bs] 
                if y is not None:
                    y_ = y[i * bs: (i+1)*bs] 
                else: 
                    y_ = None
                with pyro.plate("data"+str(i), x_.shape[0]):
                    set_weights_old(self.base_params, self.parameters_samples, self.device)
                    z = self.base_model(x_)
                    pred = pyro.deterministic("pred"+str(i), z)
                    pyro.sample("obs"+str(i), self.likelihood(pred).to_event(1), obs=y_)
        else:
            set_weights_old(self.base_params, self.parameters_samples, self.device)
            z = self.base_model(x)
            pred = pyro.deterministic("pred", z)
            pyro.sample("obs", self.likelihood(pred).to_event(1), obs=y)

    def forward(self, *args, **kwargs):
        self.parameters_samples = pyro.sample("parameters_samples", dist.Normal(torch.zeros(self.model_size), self.model_size * np.exp(self.prior_log_sigma)).to_event(1)).to(self.device)
        set_weights_old(self.base_params, self.parameters_samples, self.device)
        return self.base_model(*args, **kwargs)