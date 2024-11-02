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

MODEL = MyModel1
FN = fn1
GENERATE_DATA = generate_data1


M=2
N=100
target_log_sigma = torch.tensor(1.).log()
weights = torch.tensor([[1.0], [2.0]])   
xs, ys = GENERATE_DATA(N, M, FN, weights, target_log_sigma=target_log_sigma)
perm = torch.randperm(N)
test_train_ratio = 0.5 

x_test = xs[perm][0:int(N * test_train_ratio), :]
y_test = ys[perm][0:int(N * test_train_ratio), :]
x_train = xs[perm][int(N * test_train_ratio):, :]
y_train = ys[perm][int(N * test_train_ratio):, :]
x_train.shape, y_train.shape, x_test.shape, y_test.shape

prior_log_sigma=torch.tensor(10.).log()


################################################
# MCMC sampling

#model = MODEL(input_dim=M, output_dim=1, fn=fn, prior_log_sigma=prior_log_sigma)
model = MODEL(num_params=M, fn=FN, prior_log_sigma=prior_log_sigma)
#model = MODEL(M_inputs=M, num_hid=10, num_out=1)

likelihood_given_outputs=lambda x: dist.Normal(x, target_log_sigma.exp())

pyromodel = PyroModel(model, prior_log_sigma=prior_log_sigma,
                      likelihood_given_outputs=likelihood_given_outputs,
                      batch_size = None)

nuts_kernel_a = NUTS(pyromodel.model, step_size=1.)
mcmc_auto = MCMC(nuts_kernel_a, num_samples=1000, warmup_steps=100)
mcmc_auto.run(x_train, y_train)
mcmc_auto.summary()
posterior_samples = mcmc_auto.get_samples()['parameters_samples']



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


###################
# Laplace approximation using laplace-torch
model = MODEL(num_params=M, fn=FN, prior_log_sigma=prior_log_sigma)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss(reduction="sum")
epochs=50000
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

        param_norm = model.weights.norm()**2 / (2 * prior_log_sigma.exp()**2)
        loss_from_pred = criterion(pred, y)*N/x.shape[0] / (2 * target_log_sigma.exp()**2)
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
print(f"{MAP = }")
print(f"{posterior_samples.mean(dim=0) = }")


la = Laplace(model, likelihood="regression", hessian_structure="full", subset_of_weights='all', prior_precision=1/prior_log_sigma.exp(), sigma_noise=target_log_sigma.exp())
train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=16)
la.fit(train_loader)
print(f"{la.posterior_covariance = }")
print(f"{la.posterior_precision = }")

print(f"{la.posterior_scale = }")


print(f"{la.mean = }")
print(f"{la.H*la._H_factor = }")
print(f"{la.prior_precision_diag = }")

fig, axs = distributions_plot(posterior_samples, la.mean, la.posterior_covariance)
fig.show()


#####
# Analytic solution:
# Bayesian ML w 3 slide 5
print(f"{posterior_samples.mean(dim=0) = }")
print(f"{model.weights = }")
S = (torch.eye(M) * prior_log_sigma.exp()**-2 + target_log_sigma.exp()**-2 * FN(x_train).T @ FN(x_train)).inverse()
m = target_log_sigma.exp()**-2 * S @ FN(x_train).T @ y_train

analytical_H = S.inverse() - torch.eye(M) * prior_log_sigma.exp()**-2
fig, axs = distributions_plot(posterior_samples, m, S)
fig.show()

print(f"{m = }")
print(f"{S = }")
print(f"{S.inverse() = }")
print(f"{analytical_H = }")


################################################################################
#  LA dyi:
def make_functional_fwd(_model):
    def fn(data, parameters):
        return functional_call(_model, parameters, (data,))
    return fn

model_func = make_functional_fwd(model) # functional forward

def make_loss_func(model_func, loss_fn):
    def fn(parameters, data, target):
        param_norm = torch.nn.utils.parameters_to_vector([param for param in parameters.values()]).norm()**2 / (2 * prior_log_sigma.exp()**2)
        loss_from_pred = loss_fn(model_func(data, parameters), target) / (2*target_log_sigma.exp()**2)
        total_loss = loss_from_pred +  param_norm
        return total_loss
    return fn

model.weights.data = m.squeeze(1)
params = dict(model.named_parameters())

loss_func = make_loss_func(model_func, criterion)
gradient_fn = grad(loss_func, argnums=0)
gradient = gradient_fn(params, x_train, y_train)

hess_fn = hessian(loss_func, argnums=0)
hess = hess_fn(params, x_train, y_train) 
hess.keys()
H = hess['weights']['weights']
#precision = H*.5 + torch.eye(H.shape[0]) * prior_log_sigma.exp()**-2
precision = H
#scale = precision.inverse()
cov = precision.inverse()
print(f"{cov = }")
print(f"{precision = }")
print(f"{H = }")

fig, axs = distributions_plot(posterior_samples, params['weights'].detach(), cov.detach())
fig.show()



################################
# LA DIY with distribution based soulution

def make_functional_fwd(_model):
    def fn(data, parameters):
        return functional_call(_model, parameters, (data,))
    return fn

def make_loss_func_from_distr(model_func, prior_distribution, likelihood_given_outputs):
    def fn(parameters, data, target):
        param_vector = torch.nn.utils.parameters_to_vector([param for param in parameters.values()]).detach()
        like = likelihood_given_outputs(model_func(data, parameters)).log_prob(target).sum(dim=0)
        reg = prior_distribution.log_prob(param_vector)
        return -(like+reg)[0]
    return fn

def LA_approximation(model, x_train, y_train, prior_distribution, likelihood_given_outputs, parametersubset=None, return_hessian=True, return_gradient=True):
    if parametersubset is None:
        parametersubset = model.named_parameters()
    model_func = make_functional_fwd(model) # functional forward
    params_used = dict(parametersubset)
    loss_func = make_loss_func_from_distr(model_func, prior_distribution, likelihood_given_outputs)
    gradient_fn = grad(loss_func, argnums=0)
    return_vals = []
    if return_gradient: 
        gradient = gradient_fn(params_used, x_train, y_train)
        return_vals.append(gradient)
    if return_hessian:
        hess_fn = hessian(loss_func, argnums=0)
        hess = hess_fn(params_used, x_train, y_train)
        return_vals.append(hess)
    return return_vals

prior_distribution = torch.distributions.MultivariateNormal(torch.zeros(M), torch.eye(M)*prior_log_sigma.exp()*M)
grad_, hess_ = LA_approximation(model, x_train, y_train, prior_distribution, likelihood_given_outputs, parametersubset=None, return_hessian=True, return_gradient=True)
print(f"{grad_ = }")
print(f"{hess_ = }")















class Manifold():
    def __init__(self, model, likelihood, device="cpu", regularization=0.1, noise=0.1):
        super(Manifold, self).__init__()
        self.model = model.to(device)
        self.likelihood = likelihood
        self.device = device

        if self.likelihood == "regression":
            self.criterion = torch.nn.MSELoss()
        elif self.likelihood == "classification":
            self.criterion = torch.nn.CrossEntropyLoss()
        else:
            raise ValueError("Likelihood not recognized")
        self.regularization = regularization if type(regularization) == torch.Tensor else torch.tensor(regularization)
        self.noise = noise if type(noise) == torch.Tensor else torch.tensor(noise)

        self.MAP = None
        self.MAP_covariance = None
        self.LA = Laplace(model, likelihood="regression", hessian_structure="full", subset_of_weights='all')
        
    def set_train_data(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        self.train_loader = DataLoader(TensorDataset(self.x_train, self.y_train), batch_size=16)

    def fit(self, epochs=1, lr=0.1, optimizer=None, verbose=False, print_every_epoch=100):
        if optimizer == None:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            accum_loss = 0            
            for batch_no, (x, y) in enumerate(self.train_loader):
                optimizer.zero_grad()
                x, y = x.to(self.device), y.to(self.device)
                pred = model(x)
                loss = self.loss(x, y)
                loss.backward()
                optimizer.step()
                accum_loss += loss.item()
            if verbose:
                if epoch % print_every_epoch == 0:
                    print(f"{epoch = }, {accum_loss = }\t\t ", end="\r")
        self.MAP = torch.nn.utils.parameters_to_vector(model.parameters())
        self.LA.fit(self.train_loader)
        self.MAP_covariance = self.LA.posterior_covariance

    def set_to_map(self):
        if self.MAP is not None:
            torch.nn.utils.vector_to_parameters(self.MAP, self.model.parameters())
        else: raise ValueError("No MAP found")

    def set_weights(self, theta):
        torch.nn.utils.vector_to_parameters(theta, self.model.parameters())

    def loss_bck(self):
        return criterion(self.model(self.x_train), self.y_train)
    
    def mse_loss(self, x=None, y=None):
        if x is None:
            x = self.x_train
            y = self.y_train
        return (1.0 / self.noise) * criterion(self.model(x), y) * 0.5

    def loss(self, x=None, y=None):
        #print(f"{x.shape = }")
        #print(f"{y.shape = }")
        if x is None:
            x = self.x_train
            y = self.y_train
        return self.mse_loss(x, y) + self.regularization * torch.nn.utils.parameters_to_vector(model.parameters()).norm()**2


    def gradient_bck(self):
        #params = torch.nn.utils.parameters_to_vector(self.model.parameters())
        grads = torch.autograd.grad(self.loss(), self.model.parameters())
        return torch.cat([parm.flatten() for parm in list(grads)])

    def gradient(self):
        #params = torch.nn.utils.parameters_to_vector(self.model.parameters())
        pred = self.model(self.x_train)
        grads = torch.autograd.grad(pred, self.model.parameters())  # D times N
        grads_flat = torch.cat([parm.flatten() for parm in list(grads)])
        return 

    def covariance(self):
        self.LA.fit(self.train_loader)
        return self.LA.posterior_covariance
    
    def hess(self):
        self.LA.fit(self.train_loader)
        return la.posterior_precision

    def posterior_sample(self, n_samples=1):
        return self.LA.sample(n_samples=n_samples)
    
    def predictive_samples(self, x, n_samples=1):
        return self.LA.predictive_samples(x, n_samples=n_samples)

    # Analytic derivation of the ODE
    def ode_fun(self, t, state):
        #print(f"{state = }")
        state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
        theta = state_tensor[:self.MAP.shape[0]]
        v = state_tensor[self.MAP.shape[0]:]
        self.set_weights(theta)
        grad_val = self.gradient()
        hess_val = self.hess()
        #print(f"{grad_val = }")
        #print(f"{hess_val = }")
        #print(f"{v = }")

        acc = -(grad_val * (1 / (1 + grad_val.T @ grad_val)) * (v.T @ hess_val @ v)).flatten()
        return torch.cat([v, acc]).to("cpu").detach().numpy()
    
    def expmap(self, theta, v):
        init = torch.cat([theta, v]).to("cpu").detach().numpy()
        solution = solve_ivp(self.ode_fun, [0, 1], init, dense_output=True, rtol=1e-3, atol=1e-3)
        return solution

    def metric(self):
        D = self.MAP.shape[0]
        Id = torch.eye(D)
        Grad_val = self.gradient()
        #Hess_val = self.hess()
        M = Id + Grad_val * Grad_val.T
        return M

def make_functional_fwd(_model):
    def fn(data, parameters):
        return functional_call(_model, parameters, (data,))
    return fn

model = MyModel(num_params=M, fn=fn, prior_log_sigma=prior_log_sigma)
mymanifold = Manifold(model, likelihood="regression")
mymanifold.set_train_data(x_train, y_train)

mymanifold.fit(epochs=5000, lr=0.1, verbose=True)















#ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, 0, 0))
params = torch.nn.utils.parameters_to_vector(mymanifold.model.parameters())

gradw = ft_compute_grad(params)

grads()

grads = grad(pred, mymanifold.model.parameters())  # D times N
grads_flat = torch.cat([parm.flatten() for parm in list(grads)])




mymanifold.gradient()


init = torch.cat([mymanifold.MAP, torch.tensor([0, 0.1])]).to("cpu").detach().numpy()
solution = solve_ivp(mymanifold.ode_fun, [0, 1], init, dense_output=True, rtol=1e-10, atol=1e-10)


sol = mymanifold.expmap(mymanifold.MAP, torch.tensor([0, 0.1]))
sol.y[:,-1]
dir(sol)
sol.y
sol.t


mymanifold.LA.sample(n_samples=1).shape

mymanifold.MAP
mymanifold.MAP_covariance
mymanifold.gradient()
mymanifold.loss()
mymanifold.MAP + torch.tensor([0,0.1])

mymanifold.set_weights(mymanifold.MAP + torch.tensor([0,0.1]))
print(mymanifold.covariance())
print(mymanifold.hess().inverse())
print(mymanifold.LA.posterior_scale**2)

mymanifold.set_to_map()
print(mymanifold.covariance())
print(mymanifold.hess().inverse())

