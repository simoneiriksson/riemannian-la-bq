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
from models import MyModel, mini, PyroModel
from pandas.plotting import scatter_matrix
import pandas as pd
from pyro.infer import Predictive

def fn(xs):
    return torch.column_stack([xs[:,0], xs[:,1], xs[:,0]**2 + xs[:,1]**2])

def fn(xs):
    return xs

def generate_data(N, M, fn, weights, sigma_y=0.1):
    xs = torch.rand((N, M))*10
    #xs, _ = xs.sort(dim=0)
    ys = fn(xs) @ weights + torch.normal(torch.tensor(0.0), torch.tensor(1.)*sigma_y, size=(xs.shape[0],1))
    return xs, ys

M=2
N=100

weights = torch.tensor([[1.0], [2.0]])   
xs, ys = generate_data(N, M, fn, weights, sigma_y=1.)
perm = torch.randperm(N)
test_train_ratio = 0.5 

x_test = xs[perm][0:int(N * test_train_ratio), :]
y_test = ys[perm][0:int(N * test_train_ratio), :]
x_train = xs[perm][int(N * test_train_ratio):, :]
y_train = ys[perm][int(N * test_train_ratio):, :]
x_train.shape, y_train.shape, x_test.shape, y_test.shape

prior_log_sigma=torch.tensor(1.).log()

model = MyModel(input_dim=M, output_dim=1, fn=fn, prior_log_sigma=prior_log_sigma)
#model = mini(M_inputs=M, num_hid=10, num_out=1)
log_scale = torch.tensor(1.).log()
likelihood_given_outputs=lambda x: dist.Normal(x, log_scale.exp())

pyromodel = PyroModel(model, prior_log_sigma=prior_log_sigma,
                      likelihood_given_outputs=likelihood_given_outputs,
                      batch_size = None)

nuts_kernel_a = NUTS(pyromodel.model, step_size=1.)
mcmc_auto = MCMC(nuts_kernel_a, num_samples=1000, warmup_steps=200)
mcmc_auto.run(x_train, y_train)
mcmc_auto.summary()

# def pyromodel_manual(x, y=None):
#     D = 3
#     weights = pyro.sample("weights", dist.Normal(torch.zeros((D)), torch.ones((D))*prior_log_sigma.exp()).to_event(1))
#     pred = pyro.deterministic("pred", fn(x) @ weights)
#     with pyro.plate('data', x.shape[0]):
#         return pyro.sample('obs', likelihood_given_outputs(pred.unsqueeze(-1)).to_event(1), obs=y)


# nuts_kernel_m = NUTS(pyromodel_manual, step_size=1.)
# mcmc_manual = MCMC(nuts_kernel_m, num_samples=1000, warmup_steps=200)
# mcmc_manual.run(x_train, y_train)
# mcmc_manual.summary()


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
evals(mcmc_auto)

#print("Manual")
#evals(mcmc_manual)

posterior_samples = mcmc_auto.get_samples()['parameters_samples']

model = MyModel(input_dim=M, output_dim=1, fn=fn, prior_log_sigma=prior_log_sigma)
optimizer = torch.optim.Adam(model.parameters(), lr=.1)
criterion = torch.nn.MSELoss()

for i in range(50000):
    accum_loss = 0
    
    optimizer.zero_grad()
    x = x_train
    y = y_train
    optimizer.zero_grad()
    loss = criterion(model(x), y)
    accum_loss += loss.item()
    loss.backward()
    optimizer.step()
    if i % 100 == 0:
        print(f"{i = }, {accum_loss = }\t\t ", end="\r")


print(f"{model.weights = }")

from laplace import Laplace

la = Laplace(model, likelihood="regression", hessian_structure="full", subset_of_weights='all')
from torch.utils.data import DataLoader, TensorDataset
train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=16)
la.fit(train_loader)
print(f"{la.posterior_scale/x_train.shape[0] = }")
print(f"{la.posterior_covariance/x_train.shape[0] = }")
print(f"{la.H.inverse() = }")
print(f"{la.mean = }")


la.mean
fig, axs = distributions_plot(posterior_samples, la.mean, la.posterior_covariance)
fig.show()


###############################################################################################




mean = posterior_samples.mean(dim=0)
cov = posterior_samples.T.cov()


from pyro_utils import get_laplace
model = MyModel(input_dim=M, output_dim=1, fn=fn, prior_log_sigma=prior_log_sigma)
#model = mini(M_inputs=M, num_hid=10, num_out=1)

pyromodel = PyroModel(model, prior_log_sigma=prior_log_sigma,
                      likelihood_given_outputs=likelihood_given_outputs,
                      batch_size = None)
LA_loc, LA_cov, LA_H, delta_guide, model_guide_parameters = get_laplace(pyromodel.model, x=xs, y=ys, num_iters=10000, lr=0.1)

fig, axs = distributions_plot(posterior_samples, LA_loc=LA_loc, LA_cov=LA_cov)
fig.show()











# Collect model parameters as a list of tensors for the Hessian calculation

params = list(model.parameters())
optimizer.zero_grad()
loss = criterion(model(x_train), y_train)
J = torch.autograd.grad(loss, list(model.parameters()), create_graph=True)
J = torch.cat([e.flatten() for e in J]) # flatten
num_param = sum(p.numel() for p in model.parameters())
H = torch.zeros((num_param, num_param))
# Fill in Hessian

for i in range(num_param):
    result = torch.autograd.grad(J[i], list(model.parameters()), retain_graph=True)
    H[i] = torch.cat([r.flatten() for r in result]) # flatten
cov = H.inverse()

print(f"{model.weights = }")
print(f"{J = }")
print(f"{H = }")
print(f"{cov = }")


fig, axs = distributions_plot(posterior_samples, model.weights.data)
fig.show()