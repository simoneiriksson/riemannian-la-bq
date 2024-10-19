import pyro
import torch
import pyro.distributions as dist
import numpy as np
from torch import nn
from pyro.infer.mcmc import HMC, MCMC, NUTS
from matplotlib import pyplot as plt
import seaborn as sns
from plotting import distributions_plot
from models import MyModel, mini, PyroModel
from pandas.plotting import scatter_matrix
import pandas as pd
from pyro.infer import Predictive

def fn(xs):
    return torch.column_stack([xs[:,0], xs[:,1], xs[:,0]**2 + xs[:,1]**2])

def generate_data(xs, fn, weights, sigma_y=0.1):
    ys = fn(xs) @ weights + torch.normal(torch.tensor(0.0), torch.tensor(1.)*sigma_y, size=(xs.shape[0],1))
    return ys

M=2
N=100
xs = torch.rand((N, M))*10
xs, _ = xs.sort(dim=0)


fn(xs).shape
weights = torch.tensor([[1.0], [1.0], [1.0]])   
ys = generate_data(xs, fn, weights, sigma_y=0.01)
ys.shape

x_test = xs[0:N//5, :]
y_test = ys[0:N//5, :]
x_train = xs[N//5:, :]
y_train = ys[N//5:, :]
x_train.shape, y_train.shape, x_test.shape, y_test.shape

prior_log_sigma=torch.tensor(.1)
model = MyModel(M, fn, prior_log_sigma)
#model = mini(M_inputs=M, num_hid=10, num_out=1)
log_scale = torch.tensor(.1)
likelihood_given_outputs=lambda x: dist.Normal(x, log_scale.exp())

pyromodel = PyroModel(model, prior_log_sigma=prior_log_sigma,
                      likelihood_given_outputs=likelihood_given_outputs,
                      batch_size = 100)

nuts_kernel_a = NUTS(pyromodel.model, step_size=1.)
mcmc_auto = MCMC(nuts_kernel_a, num_samples=200, warmup_steps=200)
mcmc_auto.run(x_train, y_train)
mcmc_auto.summary()

def pyromodel_manual(x, y=None):
    D = 3
    weights = pyro.sample("weights", dist.Normal(torch.zeros((D)), torch.ones((D))*prior_log_sigma.exp()).to_event(1))
    pred = pyro.deterministic("pred", fn(x) @ weights)
    with pyro.plate('data', x.shape[0]):
        return pyro.sample('obs', likelihood_given_outputs(pred.unsqueeze(-1)).to_event(1), obs=y)


nuts_kernel_m = NUTS(pyromodel_manual, step_size=1.)
mcmc_manual = MCMC(nuts_kernel_m, num_samples=200, warmup_steps=200)
mcmc_manual.run(x_train, y_train)
mcmc_manual.summary()


def evals(mcmc):
    samples = mcmc.get_samples()
    for param in samples.keys():
        print(f"param = {param}")
        print(f"posterior means = {samples[param].mean(dim=0)}")
        print(f"posterior std = {samples[param].std(dim=0)}")
        print("\n") 

    df = pd.DataFrame(samples[param])
    scatter_matrix(df, alpha = 0.2, figsize = (6, 6), diagonal = 'kde')

print("Auto")
evals(mcmc_auto)

print("Manual")
evals(mcmc_manual)

meh = Predictive(pyromodel_manual, mcmc_manual.get_samples())(x_train)
plt.scatter(x_train[:,1], y_train)
plt.scatter(x_train[:,0], y_train)
plt.scatter(x_train[:,0], meh["y_pred"].mean(dim=0))


meh2 = Predictive(pyromodel.model, mcmc_auto.get_samples())(x_train)
meh.keys()