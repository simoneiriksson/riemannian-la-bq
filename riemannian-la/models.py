import torch
from torch import nn
from utils import extract_parameters, set_weights_old
from pyro import distributions as dist
import numpy as np
import pyro
from utils import get_parameter_vector_indices

class MyModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim, fn, prior_log_sigma):
        super(MyModel, self).__init__()
        self.input_dim = input_dim
        dummy = torch.zeros((1, input_dim))
        self.num_params = fn(dummy).shape[1]
        self.output_dim = output_dim
        self.weights = torch.nn.Parameter(torch.rand((self.num_params, output_dim))*prior_log_sigma)
        #self.weights = torch.nn.Parameter(torch.rand((3,1))*prior_log_sigma)
        self.fn = fn

    def forward(self, xs):
        return self.fn(xs) @ self.weights


class MyModel_stoch(torch.nn.Module):
    def __init__(self, input_dim, fn, prior_log_sigma):
        super(MyModel_stoch, self).__init__()
        self.input_dim = input_dim
        dummy = torch.zeros((1, input_dim))
        self.num_params = fn(dummy).shape[1]
        #self.output_dim = output_dim
        self.weights = torch.nn.Parameter(torch.rand((self.num_params))*prior_log_sigma)
        self.fn = fn

    def forward(self, xs):
        print(f"{self.weights.shape}")
        print(f"{self.fn(xs).shape}")
        return self.fn(xs) * self.weights.unsqueeze(0)




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

    def model(self, x=None, y=None):
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
            shape = x.shape[0] if x is not None else 1
            with pyro.plate("data", shape):
                set_weights_old(self.base_params, self.parameters_samples, self.device)
                if x is not None:
                    z = self.base_model(x)
                else:    
                    z = self.base_model()
                pred = pyro.deterministic("pred", z)
                pyro.sample("obs", self.likelihood(pred).to_event(1), obs=y)

    def forward(self, *args, **kwargs):
        #self.parameters_samples = pyro.sample("parameters_samples", dist.Normal(torch.zeros(self.model_size), self.model_size * np.exp(self.prior_log_sigma)).to_event(1)).to(self.device)
        #set_weights_old(self.base_params, self.parameters_samples, self.device)
        return self.base_model(*args, **kwargs)
    

# Adapted from https://github.com/wjmaddox/drbayes/blob/master/subspace_inference/posteriors/pyro.py
class PyroModel2(torch.nn.Module):
    def __init__(self, base, prior_log_sigma=None, likelihood_given_outputs=None, batch_size = None, device="cpu", *args, **kwargs):
        super(PyroModel2, self).__init__()
        self.base_model = base
        self.params = []
        for name, module in self.base_model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                self.params.append((module, param_name, param.size()))

        self.base_params = self.params

        #self.rank = cov_factor.size()[0]
        self.prior_log_sigma = prior_log_sigma
        
        self.likelihood = likelihood_given_outputs
        self.batch_size = batch_size
        self.model_size = sum([p[2].numel() for p in self.base_params])
        self.device = device

    def model(self, x=None, y=None):
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
                    offset = 0
                    for module, name, shape in self.params:
                        size = np.prod(shape)	       
                        value = self.parameters_samples[offset:offset + size]
                        setattr(module, name, value.view(shape).to(self.device))	
                        offset += size
                    #set_weights_old(self.base_params, self.parameters_samples, self.device)
                    z = self.base_model(x_)
                    pred = pyro.deterministic("pred"+str(i), z)
                    pyro.sample("obs"+str(i), self.likelihood(pred).to_event(1), obs=y_)
        else:
            shape = x.shape[0] if x is not None else 1
            with pyro.plate("data", shape):
                offset = 0
                for module, name, shape in self.params:
                    size = np.prod(shape)	       
                    value = self.parameters_samples[offset:offset + size]
                    #torch.nn.utils.vector_to_parameters(value.view(shape), module._parameters[name])
                    setattr(module, name, value.view(shape).to(self.device))
                    offset += size

                #set_weights_old(self.base_params, self.parameters_samples, self.device)
                if x is not None:
                    z = self.base_model(x)
                else:    
                    z = self.base_model()
                pred = pyro.deterministic("pred", z)
                pyro.sample("obs", self.likelihood(pred).to_event(1), obs=y)

    def forward(self, *args, **kwargs):
        #self.parameters_samples = pyro.sample("parameters_samples", dist.Normal(torch.zeros(self.model_size), self.model_size * np.exp(self.prior_log_sigma)).to_event(1)).to(self.device)
        #set_weights_old(self.base_params, self.parameters_samples, self.device)
        print(f"{args = }")
        print(f"{kwargs = }")
        return self.base_model(*args, **kwargs)
    


# Adapted from https://github.com/wjmaddox/drbayes/blob/master/subspace_inference/posteriors/pyro.py
class PyroModel_subset(torch.nn.Module):
    def __init__(self, base, prior_log_sigma, likelihood_given_outputs, parametersubset=None, batch_size = None, device="cpu", *args, **kwargs):
        super().__init__()
        self.base_model = base
        self.base_params = extract_parameters(self.base_model)

        #self.rank = cov_factor.size()[0]
        self.prior_log_sigma = prior_log_sigma
        
        self.likelihood = likelihood_given_outputs
        self.batch_size = batch_size
        self.device = device
        if parametersubset is None:
            parametersubset = self.base_model.named_parameters()
            num_params =  sum([p[1].numel() for p in self.base_model.named_parameters()])
        else:
            num_params =  sum([p.numel() for p in parametersubset.values()])
        self.params_used = dict(parametersubset)
        self.num_params = num_params
        self.parameter_subset_indices = get_parameter_vector_indices(self.base_model, self.params_used.keys())
        #self.fixed_params_indices = torch.tensor(set(range(self.num_params)) - set(self.parameter_subset_indices))
        
    def model(self, x=None, y=None):
        self.parameters_samples_subset = pyro.sample("parameters_samples", 
                                              dist.Normal(torch.zeros(self.num_params), 
                                                          torch.ones(self.num_params) * np.exp(self.prior_log_sigma)).to_event(1)).to(self.device)
        
        set_weights_old(self.params_used, self.parameters_samples_subset, self.device)
        
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
            shape = x.shape[0] if x is not None else 1
            with pyro.plate("data", shape):
                set_weights_old(self.base_params, self.parameters_samples, self.device)
                if x is not None:
                    z = self.base_model(x)
                else:    
                    z = self.base_model()
                pred = pyro.deterministic("pred", z)
                pyro.sample("obs", self.likelihood(pred).to_event(1), obs=y)

    def forward(self, *args, **kwargs):
        #self.parameters_samples = pyro.sample("parameters_samples", dist.Normal(torch.zeros(self.model_size), self.model_size * np.exp(self.prior_log_sigma)).to_event(1)).to(self.device)
        #set_weights_old(self.base_params, self.parameters_samples, self.device)
        return self.base_model(*args, **kwargs)