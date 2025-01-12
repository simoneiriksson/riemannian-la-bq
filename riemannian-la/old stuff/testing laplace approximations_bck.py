# This code implements and test a class that wraps around a torch model and gives a pyro model function.
from laplace_approx import hessian_dict_to_matrix

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
from scipy.integrate import solve_ivp

from utils import set_weights_old, extract_parameters, set_weights
from laplace import Laplace
from torch.utils.data import DataLoader, TensorDataset
from torch.func import grad, jvp, vjp, hessian, jacfwd, jacrev, vmap, functional_call

def fn1(xs):
    return torch.column_stack([xs[:,0], xs[:,1]])

def generate_data1(N, M, fn, weights, target_log_sigma=0.1):
    xs = torch.rand((N, M))*10
    #xs = torch.ones(N, M)
    #xs, _ = xs.sort(dim=0)
    ys = fn(xs) @ weights + torch.normal(torch.tensor(0.0), torch.tensor(1.)*torch.tensor(target_log_sigma).exp(), size=(xs.shape[0],1))
    #ys = torch.normal(torch.tensor(0.0), torch.tensor(1.)*torch.tensor(log_scale).exp(), size=(xs.shape[0],1))
    return xs, ys

def fn2(xs):
    return xs

def generate_data2(N, M, fn, weights, target_log_sigma=0.1):
    #xs = torch.rand((N, M))*10
    xs = torch.ones(N, M)
    #xs, _ = xs.sort(dim=0)
    #ys = fn(xs) @ weights + torch.normal(torch.tensor(0.0), torch.tensor(1.)*torch.tensor(log_scale).exp(), size=(xs.shape[0],1))
    ys = torch.normal(torch.tensor(0.0), torch.tensor(1.)*torch.tensor(target_log_sigma).exp(), size=(xs.shape[0],1))
    return xs, ys

class MyModel2(torch.nn.Module):
    def __init__(self, num_params, fn, prior_log_sigma):
        super(MyModel2, self).__init__()
        self.num_params = num_params
        self.weights = torch.nn.Parameter(torch.rand((self.num_params))*prior_log_sigma)
        self.paws = torch.arange(1, (num_params*2)+1, 2)
        self.fn = fn
    def forward(self, xs):
        if xs.dim() == 1:
            xs = xs.unsqueeze(0)
            return  ((self.fn(xs) * self.weights**self.paws).sum(dim=1) - 1).squeeze(-1)
        else:
            return  ((self.fn(xs) * self.weights**self.paws).sum(dim=1) - 1).unsqueeze(-1)


class MyModel1(torch.nn.Module):
    def __init__(self, num_params, fn, prior_log_sigma):
        super(MyModel1, self).__init__()
        self.num_params = num_params
        self.weights = torch.nn.Parameter(torch.rand((self.num_params))*prior_log_sigma)
        self.paws = torch.arange(1, (num_params*2)+1, 2)
        self.fn = fn
    def forward(self, xs):
        if xs.dim() == 1:
            xs = xs.unsqueeze(0)
            return  ((self.fn(xs) @ self.weights))
        else:
            return  ((self.fn(xs) @ self.weights)).unsqueeze(-1)


class MyModel3(torch.nn.Module):
    def __init__(self, num_params, fn, prior_log_sigma):
        super().__init__()
        self.num_params = num_params
        self.fc1 = nn.Linear(num_params, 1, bias=False)
        self.fc2 = nn.Linear(1, 1, bias=False)
        
    def forward(self, xs):
        return self.fc2(torch.relu(self.fc1(xs))).squeeze(-1)

def generate_data_like_model(model_class, N, M, fn,target_log_sigma=0.1, prior_log_sigma=0.1, weights=None):
    xs = torch.rand((N, M))*10
    ys = torch.zeros((N, 1))
    if type(target_log_sigma) != torch.Tensor: target_log_sigma = torch.tensor(target_log_sigma)
    if type(prior_log_sigma)  != torch.Tensor: prior_log_sigma = torch.tensor(prior_log_sigma)
    model = model_class(num_params=M, fn=FN, prior_log_sigma=prior_log_sigma)
    model_size = sum([p.numel() for p in model.parameters()])
    if weights is None:
        weights = torch.distributions.Normal(0, prior_log_sigma.exp()).sample((model_size, 1))
    
    for i in range(N):
        sample_weights = torch.distributions.Normal(weights, prior_log_sigma.exp()).sample()
        torch.nn.utils.vector_to_parameters(sample_weights, model.parameters())
        #print(f"{model.weights = }")
        y = model(xs[i]) + torch.normal(torch.tensor(0.0), target_log_sigma.exp())
        #print(f"{y.shape = }")
        ys[i,0] = y
    print(f"{weights = }")
    return xs.clone().detach(), ys.clone().detach()


# With these settings, I get something that looks right
MODEL = MyModel1
FN = fn1
GENERATE_DATA = generate_data1

# With these settings, there is a difference between laplace-torch and my own implementation
MODEL = MyModel3
FN = fn1
GENERATE_DATA = generate_data1

M=2
N=100
target_log_sigma = torch.tensor(1.).log()
weights = torch.tensor([[1.0], [2.0]])   
weights = torch.tensor([[-1.1422],[-0.5213],[-1.1982]])
prior_log_sigma=torch.tensor(1.).log()
#xs, ys = GENERATE_DATA(N, M, FN, weights, target_log_sigma=target_log_sigma)
xs, ys = generate_data_like_model(MODEL, N, M, FN, target_log_sigma=target_log_sigma.clone().detach(), prior_log_sigma=prior_log_sigma.clone().detach(), weights=weights)
perm = torch.randperm(N)
test_train_ratio = 0.5 

x_test = xs[perm][0:int(N * test_train_ratio), :]
y_test = ys[perm][0:int(N * test_train_ratio), :]
x_train = xs[perm][int(N * test_train_ratio):, :]
y_train = ys[perm][int(N * test_train_ratio):, :]
x_train.shape, y_train.shape, x_test.shape, y_test.shape


################################################
# MCMC sampling

#model = MODEL(input_dim=M, output_dim=1, fn=fn, prior_log_sigma=prior_log_sigma)
model_for_pyro = MODEL(num_params=M, fn=FN, prior_log_sigma=prior_log_sigma)
num_params =  sum([p[1].numel() for p in model_for_pyro.named_parameters()])
print(f"model has {num_params} parameters ")

#model = MODEL(M_inputs=M, num_hid=10, num_out=1)

likelihood_given_outputs=lambda x: dist.Normal(x, target_log_sigma.exp())

pyromodel = PyroModel(model_for_pyro, prior_log_sigma=prior_log_sigma,
                      likelihood_given_outputs=likelihood_given_outputs,
                      batch_size = None)

nuts_kernel_a = NUTS(pyromodel.model, step_size=1.)
mcmc_auto = MCMC(nuts_kernel_a, num_samples=1000, warmup_steps=100)
mcmc_auto.run(x_train, y_train)
mcmc_auto.summary()
posterior_samples = mcmc_auto.get_samples()['parameters_samples']


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


###################
# Finding MAP solution:
model = MODEL(num_params=M, fn=FN, prior_log_sigma=prior_log_sigma)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.MSELoss(reduction="sum")
epochs=10000
train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=500)
for epoch in range(epochs):
    accum_loss = 0            
    for batch_no, (x, y) in enumerate(train_loader):
        #print(f"{x.shape = }")
        optimizer.zero_grad()
        pred = model(x)
        #loss = 1/x.shape[0] * criterion(pred, y)/target_log_sigma.exp()**2 + prior_log_sigma.exp()**-1 * model.weights.norm()**2
        #loss = -likelihood_given_outputs(pred).log_prob(y).sum()
        #mse_loss = criterion(pred, y) / (2 * target_log_sigma.exp()**2)
        #reg_loss = prior_log_sigma.exp()**-1 * model.weights.norm()**2 / 2
        #loss = mse_loss + reg_loss

        param_norm = torch.nn.utils.parameters_to_vector(model.parameters()).norm()**2 / (2 * prior_log_sigma.exp()**2)
        loss_from_pred = criterion(pred, y)*x_train.shape[0]/x.shape[0] / (2 * target_log_sigma.exp()**2)
        loss = loss_from_pred + param_norm
        #prior_prob = torch.distributions.Normal(torch.zeros((M)), torch.ones((M))*prior_log_sigma.exp()).log_prob(model.weights).sum()
        #like = likelihood_given_outputs(pred).log_prob(y).sum()
        #loss = -(like + prior_prob)
        loss.backward()
        optimizer.step()
        accum_loss += loss.item()
    if epoch % 100 == 0:
        print(f"{epoch = }, {accum_loss = }\t\t ", end="\r")
        pass
MAP = torch.nn.utils.parameters_to_vector(model.parameters())

print([p for p in model.parameters()])
print(f"{posterior_samples.mean(dim=0) = }")

# Laplace approximation using diy laplace
from laplace_approx import LA_approximation
grad_, hess_ = LA_approximation(model, xs=x_train, ys=y_train, task_type="regression", prior_sigma=prior_log_sigma.exp(), target_sigma=target_log_sigma.exp())
hess_matrix, parameter_properties = hessian_dict_to_matrix(hess_)
hess_size=hess_matrix.shape[0]
print(f"{hess_matrix = }")

my_posterior_precision = hess_matrix + torch.eye(hess_size)*1/prior_log_sigma.exp()
my_covariance_matrix = torch.inverse(my_posterior_precision)
print(f"{hess_matrix + torch.eye(hess_size)*1/prior_log_sigma.exp() = }")
fig, axs = distributions_plot(posterior_samples, MAP.detach(), my_covariance_matrix.detach())
fig.show()



# Laplace approximation using laplace-torch
la = Laplace(model, likelihood="regression", hessian_structure="full", subset_of_weights='all', prior_precision=1/prior_log_sigma.exp(), sigma_noise=target_log_sigma.exp())
train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=16)
la.fit(train_loader)
print(f"{la.posterior_covariance = }")
print(f"{la.posterior_precision = }")

print(f"{la.posterior_scale = }")

print(f"{la.mean = }")
print(f"{la.H*la._H_factor + torch.diag(la.prior_precision_diag)= }")
print(f"{la.prior_precision_diag = }")

fig, axs = distributions_plot(posterior_samples, la.mean, la.posterior_covariance)
fig.show()


torch.isclose(la.posterior_precision, my_posterior_precision, atol=1e-2).all()
la_posterior_precision = la.posterior_precision
diff = la.posterior_precision - my_posterior_precision






