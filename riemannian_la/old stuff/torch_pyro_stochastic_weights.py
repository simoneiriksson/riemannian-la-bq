# This code implements and test a class that wraps around a torch model and gives a pyro model function.

import pyro
import torch
import pyro.distributions as dist
import numpy as np
from torch import nn
from pyro.infer.mcmc import HMC, MCMC, NUTS
from matplotlib import pyplot as plt
import seaborn as sns
from plotting import distributions_plot
from models import MyModel, mini, PyroModel, MyModel_stoch
from pandas.plotting import scatter_matrix
import pandas as pd
from pyro.infer import Predictive
import torch.distributions as torch_d
from pyro_utils import get_laplace
from laplace import Laplace

prior_log_sigma=torch.tensor(1.).log()
log_scale = torch.tensor(.05).log()
likelihood_given_outputs=lambda x: dist.Normal(x, log_scale.exp())

D = 3
paws = torch.arange(1, D+1)

class MyModel(torch.nn.Module):
    def __init__(self, num_params, prior_log_sigma):
        super(MyModel, self).__init__()
        self.num_params = num_params
        self.weights = torch.nn.Parameter(torch.rand((self.num_params))*prior_log_sigma)
        self.paws = torch.arange(1, num_params+1)
    def forward(self):
        return (self.weights**self.paws).sum().unsqueeze(-1)

model  = MyModel(num_params=D, prior_log_sigma=prior_log_sigma)

pyromodel = PyroModel(model, prior_log_sigma=prior_log_sigma,
                      likelihood_given_outputs=likelihood_given_outputs,
                      batch_size = None)


def pyromodel_manual(y=None):
    weights = pyro.sample("parameters_samples", dist.Normal(torch.zeros((D)), torch.ones((D))*prior_log_sigma.exp()).to_event(1))
    pred = pyro.deterministic("pred", (weights**paws).sum()).unsqueeze(-1)
    #pred = pyro.deterministic("pred", (weights*torch.pi).sin().sum()).unsqueeze(-1)
    with pyro.plate('data'):
        return pyro.sample('obs', likelihood_given_outputs(pred).to_event(1), obs=torch.tensor(0.0))

#infermodel = pyromodel_manual

infermodel = pyromodel.model

nuts_kernel = NUTS(infermodel, step_size=1.)
mcmc = MCMC(nuts_kernel, num_samples=1000, warmup_steps=10)
mcmc.run(y=torch.tensor(0.0))
mcmc.summary()

posterior_samples = mcmc.get_samples()

posterior_samples.keys()

def evals(mcmc):
    samples = mcmc.get_samples()
    for param in samples.keys():
        print(f"param = {param}")
        print(f"posterior means = {samples[param].mean(dim=0)}")
        print(f"posterior std = {samples[param].std(dim=0)}")
        print("\n") 

        df = pd.DataFrame(samples[param])
        scatter_matrix(df, alpha = 0.2, figsize = (6, 6), diagonal = 'kde')
        plt.show()
print("Auto")
evals(mcmc)


LA_loc, LA_cov, LA_H, delta_guide, model_guide_parameters = get_laplace(infermodel, x=None, y=torch.tensor(0.0), num_iters=5000, lr=0.01)


# collect samples in one tensor, in the same order as the laplace approximation
posterior_samples_list = []
posterior_samples_names =[]
for parameter, size in model_guide_parameters.items():
    if parameter in posterior_samples:
        posterior_samples_names += [f"{parameter}_{i}" for i in range(size.numel())]
        posterior_samples_list.append(posterior_samples[parameter].reshape(-1, *size))


posterior_samples_tensor = torch.cat(posterior_samples_list, dim=-1)

print(f"{posterior_samples_tensor.mean(dim=0) = }, \n{posterior_samples_tensor.std(dim=0)**2 = }")
print(f"{LA_loc = }, \n{LA_cov.diag() = }")

print(f"{posterior_samples_tensor.T.cov() = }")
print(f"{LA_cov = }")


fig, axs = distributions_plot(posterior_samples_tensor, LA_loc, LA_cov)
fig.show()


#################################################################################
# Try out Riemannian Laplace approximation
# Sample from LA
# make a function that returns metric at a point
# magically feed this into scipy shooter

# get hessian at MAP

def jacobian(y, x, create_graph=False):                                                               
    jac = []                                                                                          
    flat_y = y.reshape(-1)                                                                            
    grad_y = torch.zeros_like(flat_y)                                                                 
    for i in range(len(flat_y)):                                                                      
        grad_y[i] = 1.                                                                                
        grad_x, = torch.autograd.grad(flat_y, x, grad_y, retain_graph=True, create_graph=create_graph)
        jac.append(grad_x.reshape(x.shape))                                                           
        grad_y[i] = 0.                                                                                
    return torch.stack(jac).reshape(y.shape + x.shape)                   

from utils import extract_parameters, set_weights_old

model  = MyModel(num_params=D, prior_log_sigma=prior_log_sigma)

set_weights_old(extract_parameters(model), LA_loc, pyromodel.device)

pred = model()


log_prob = likelihood_given_outputs(pred).to_event(1).log_prob(torch.tensor(0.0).unsqueeze(0).unsqueeze(0))

log_prob.requires_grad_(True)

from get_hessian import hessian, jacobian
jacobian(log_prob, extract_parameters(model)).shape



sample = LA_loc + torch.randn_like(LA_loc) @ torch.linalg.cholesky(LA_cov)

sample_dict = {'parameters_samples': sample.unsqueeze(0)}

set_weights_old(pyromodel.base_params, sample, pyromodel.device)
pyromodel.base_params[0][0].weights

pred = pyromodel()
log_prob = pyromodel.likelihood(pred).to_event(1).log_prob(torch.tensor(0.0).unsqueeze(0).unsqueeze(0))
log_prob.backward()

def metric_at_point(point):
    pyromodel_manual
    