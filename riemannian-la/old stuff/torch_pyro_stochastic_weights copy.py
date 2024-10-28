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

def fn(xs):
    return torch.column_stack([xs[:,0], xs[:,1], xs[:,0]**2 + xs[:,1]**2])

def fn(xs):
    return torch.column_stack([torch.ones_like(xs[:,0])]*M)


def fn(xs):
    return torch.column_stack([xs[:,0], xs[:,1]])

def fn(xs):
    return torch.column_stack([torch.ones_like(xs[:,0])]*M)

def fn(xs):
    return torch.column_stack([xs[:,0], xs[:,1]])

def fn(xs):
    return xs


def generate_data(N, M, fn, sigma_y=0.1):
    xs = torch.rand((N, M))*10
    xs, _ = xs.sort(dim=0)

    N_weight_samples = N
    weights_dim = fn(xs[0:1]).shape[1]
    number_mix = 5
    mus = torch.row_stack([torch.ones(weights_dim)*i for i in range(number_mix)])
    Sigmas = torch.cat([torch.eye(weights_dim).unsqueeze(0)*.1] * mus.shape[0])
    mix = torch_d.Categorical(torch.ones(number_mix,))
    comp = torch_d.MultivariateNormal(mus, Sigmas)
    gmm = torch_d.MixtureSameFamily(mix, comp)
    weights = gmm.sample(sample_shape=torch.Size([N_weight_samples]))
    ys = (fn(xs)* weights).sum(dim=1) + torch.normal(0, sigma_y, size=(xs.shape[0],1))
    return xs, ys, weights

def generate_data(N, M, sigma_y=0.1):
    xs = torch.rand((N, M))*10
    #xs = torch.ones(N, M)
    #xs, _ = xs.sort(dim=0)
    weights_dim = fn(xs[0:1,:]).shape[1]
    paws = torch.arange(1, weights_dim+1)
    #ts = torch.linspace(0,torch.pi, N)
    ts = torch.linspace(0, 1, N)
    #mus = torch.column_stack([torch.cos(ts), torch.sin(ts)])
    mus = torch.column_stack([ts, ts**2])
    Sigmas = torch.cat([torch.eye(weights_dim).unsqueeze(0)*.0001] * mus.shape[0])
    weights = torch_d.MultivariateNormal(mus, Sigmas).sample()
    #ys = (fn(xs) * weights**paws).sum(dim=1).unsqueeze(-1) + torch.normal(0, sigma_y, size=(xs.shape[0], ))
    #ys = (xs[:,0] * weights[:,0] + xs[:,1] * weights[:,1]**2) + torch.normal(0, sigma_y, size=(xs.shape[0], ))
    #print(f"{(xs * weights).shape}")
    ys = (xs * weights).sum(dim=1) + torch.normal(0, sigma_y, size=(xs.shape[0], ))
    
    return xs, ys.unsqueeze(1), weights


def generate_data(N, M, sigma_y=0.1):
    xs = torch.rand((N, M))*10
    ys = xs.sum(dim=1).unsqueeze(1) + torch.normal(0, sigma_y, size=(xs.shape[0],1))
    return xs, ys


M=2
N=100

xs, ys = generate_data(N,M, sigma_y=1.0)
print(f"{xs.shape = }, {ys.shape = }")
perm = torch.randperm(N)

plt.scatter(x=xs[:,0], y=xs[:,1], c=ys[:,0])
plt.show()

test_train_ratio = 0.5 

x_test = xs[perm][0:int(N * test_train_ratio), :]
y_test = ys[perm][0:int(N * test_train_ratio), :]
x_train = xs[perm][int(N * test_train_ratio):, :]
y_train = ys[perm][int(N * test_train_ratio):, :]
x_train.shape, y_train.shape, x_test.shape, y_test.shape

class MyModel_stoch(torch.nn.Module):
    def __init__(self, num_params, fn, prior_log_sigma):
        super(MyModel_stoch, self).__init__()
        self.num_params = num_params
        self.weights = torch.nn.Parameter(torch.rand((self.num_params))*prior_log_sigma)
        self.paws = torch.arange(1, num_params+1)
        self.fn = fn
    def forward(self, xs):
        return xs.sum(dim=1).unsqueeze(1) + (self.weights**self.paws).sum(dim=0).unsqueeze(-1)-1


prior_log_sigma=torch.tensor(1.).log()
log_scale = torch.tensor(1.0).log()
likelihood_given_outputs=lambda x: dist.Normal(x, log_scale.exp())

model = MyModel_stoch(num_params=M, fn=fn, prior_log_sigma=prior_log_sigma)

pred =  model(x_train)
pred.shape

pyromodel = PyroModel(model, prior_log_sigma=prior_log_sigma,
                      likelihood_given_outputs=likelihood_given_outputs,
                      batch_size = None)

infermodel = pyromodel.model

D = M
paws = torch.arange(1, D+1)
paws = torch.tensor([1,2])

def pyromodel_manual(xs, ys=None):
    weights = pyro.sample("parameters_samples", dist.Normal(torch.zeros((D)), torch.ones((D))*prior_log_sigma.exp()).to_event(1))
    pred = pyro.deterministic("pred", xs.sum(dim=1).unsqueeze(1) + ((weights**paws).sum(-1)).unsqueeze(-1)-1)
    #pred = pyro.deterministic("pred", xs.sum(dim=1).unsqueeze(1) + weights[0].sin().unsqueeze(-1)**2 + weights[1].cos().unsqueeze(-1)**2 - 1)
    # print(f"{fn(xs).shape = } ")
    # print(f"{weights.shape = }")
    # print(f"{paws.shape = }")
    # print(f"{(xs * weights**paws).shape = }")
    # print(f"{(xs * weights**paws).sum(-1).shape = }")
    # print(f"{pred.shape = }")
    #pred = pyro.deterministic("pred", (weights*torch.pi).sin().sum()).unsqueeze(-1)
    with pyro.plate('data'):
        return pyro.sample('obs', likelihood_given_outputs(pred).to_event(1), obs=ys)

infermodel = pyromodel_manual
pred =  infermodel(x_train)

nuts_kernel = NUTS(infermodel, step_size=1.)
mcmc = MCMC(nuts_kernel, num_samples=1000, warmup_steps=10)
mcmc.run(x_train, y_train)
mcmc.summary()


def evals(mcmc):
    samples = mcmc.get_samples()
    for param in samples.keys():
        print(f"param = {param}")
        print(f"posterior means = {samples[param].mean(dim=0)}")
        print(f"posterior std = {samples[param].std(dim=0)}")
        print("\n") 

        df = pd.DataFrame(samples[param])
        #df['mult'] = pd.DataFrame(samples[param][:, 0] - samples[param][:, 1]**2)
        scatter_matrix(df, alpha = 0.2, figsize = (6, 6), diagonal = 'kde')
        plt.show()
print("Auto")
evals(mcmc)

from pyro_utils import get_laplace
from laplace import Laplace

posterior_samples = mcmc.get_samples()

LA_loc, LA_cov, LA_H, delta_guide, model_guide_parameters = get_laplace(infermodel, x=xs, y=ys, num_iters=5000, lr=0.001)

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

    
model = MyModel_stoch(num_params=M, fn=fn, prior_log_sigma=prior_log_sigma)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
criterion = torch.nn.MSELoss()

for i in range(5):
    accum_loss = 0
    for i in range(x_train.shape[0]):
        optimizer.zero_grad()
        x = x_train[i:i+1]
        y = y_train[i]
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        accum_loss += loss.item()
        loss.backward()
        optimizer.step()
    print(f"{accum_loss = }")

for i in range(50):
    accum_loss = 0
    
    optimizer.zero_grad()
    x = x_train
    y = y_train
    optimizer.zero_grad()
    loss = criterion(model(x), y)
    accum_loss += loss.item()
    loss.backward()
    optimizer.step()
    print(f"{accum_loss = }")


print(f"{model.weights = }")

# Define a helper function to compute the loss at a given point
def model_loss_fn(params):
    model.eval()
    with torch.no_grad():
        # Update model parameters with provided params
        start = 0
        for param in model.parameters():
            param_shape = param.size()
            numel = param.numel()
            param.copy_(params[start:start + numel].view(param_shape))
            start += numel
        
        # Forward pass to calculate loss
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
    return loss

# Flatten the model parameters to pass them to the Hessian function
params = torch.cat([param.flatten() for param in model.parameters()])

# Compute the Hessian of the loss with respect to the parameters
from torch.autograd.functional import hessian
from torch.autograd.functional import jacobian
hess_matrix = hessian(model_loss_fn, params)
jacob = jacobian(model_loss_fn, params)












# Define a function that returns the loss, given the model parameters as input
def loss_fn(params):
    # Load params back into the model
    with torch.no_grad():
        for p, param in zip(params, model.parameters()):
            param.copy_(p)
    
    # Forward pass and loss computation
    output = model(x_train)
    loss = criterion(output, y_train)
    return loss

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


params_fixed = [param.clone() for param in model.parameters()]
params_fixed = torch.cat([e.flatten() for e in params_fixed]) # flatten

hess = torch.autograd.functional.hessian(loss_fn, params_fixed)