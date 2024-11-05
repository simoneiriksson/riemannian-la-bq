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
        #self.paws = torch.arange(1, (num_params*2)+1, 2)
        self.paws = torch.arange(0, num_params-1, 1)
        self.fn = fn
    def forward(self, xs):
        if xs.dim() == 1:
            xs = xs.unsqueeze(0)
            return  ((self.fn(xs) * self.weights**self.paws).sum(dim=1) - 1).squeeze(-1)
        else:
            return  ((self.fn(xs) * self.weights**self.paws).sum(dim=1) - 1).unsqueeze(-1)


class BananaModel(torch.nn.Module):
    def __init__(self, num_params, fn, prior_log_sigma):
        super().__init__()
        self.num_params = num_params
        self.weights = torch.nn.Parameter(torch.rand((self.num_params))*prior_log_sigma)
        #self.paws = torch.arange(1, (num_params*2)+1, 2)
        self.paws = torch.arange(1, num_params+1, 1)
        self.fn = fn
    def forward(self, xs):
        if xs.dim() == 1:
            xs = xs.unsqueeze(0)
            return  ((self.weights**self.paws).sum(dim=0) - 1).squeeze(-1)
        else:
            return  ((self.weights**self.paws).sum(dim=0) - 1).unsqueeze(-1)



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
        self.fc2 = nn.Linear(1, 1)
        
    def forward(self, xs):
        return self.fc2(torch.relu(self.fc1(xs))).squeeze(-1)

MODEL = BananaModel
FN = fn1
GENERATE_DATA = generate_data2


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

prior_log_sigma=torch.tensor(1.).log()

################################################
# MCMC sampling

#model = MODEL(input_dim=M, output_dim=1, fn=fn, prior_log_sigma=prior_log_sigma)
model_for_pyro = MODEL(num_params=M, fn=FN, prior_log_sigma=prior_log_sigma)
model_for_pyro.forward(x_train)
model_for_pyro.weights
model_for_pyro.paws



num_params =  sum([p[1].numel() for p in model_for_pyro.named_parameters()])
print(f"model has {num_params} parameters ")

#model = MODEL(M_inputs=M, num_hid=10, num_out=1)

likelihood_given_outputs=lambda x: dist.Normal(x, target_log_sigma.exp())

pyromodel = PyroModel(model_for_pyro, prior_log_sigma=prior_log_sigma,
                      likelihood_given_outputs=likelihood_given_outputs,
                      batch_size = None)

nuts_kernel_a = NUTS(pyromodel.model, step_size=1.)
mcmc_auto = MCMC(nuts_kernel_a, num_samples=500, warmup_steps=100)
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
        loss = loss_from_pred #+ param_norm
        #prior_prob = torch.distributions.Normal(torch.zeros((M)), torch.ones((M))*prior_log_sigma.exp()).log_prob(model.weights).sum()
        #like = likelihood_given_outputs(pred).log_prob(y).sum()
        #loss = -(like + prior_prob)
        loss.backward()
        #[p.grad for p in model.parameters()]
        optimizer.step()
        accum_loss += loss.item()
    if epoch % 100 == 0:
        print(f"{epoch = }, {accum_loss = }\t\t ", end="\r")
        pass
MAP = torch.nn.utils.parameters_to_vector(model.parameters())

print([p for p in model.parameters()])
posterior_samples.mean(dim=0)


# Laplace approximation using diy laplace
from laplace_approx import LA_approximation
grad_, hess_ = LA_approximation(model, xs=x_train, ys=y_train, task_type="regression", prior_sigma=prior_log_sigma.exp(), target_sigma=target_log_sigma.exp())
hess_matrix, parameter_properties = hessian_dict_to_matrix(hess_)
hess_size=hess_matrix.shape[0]
print(f"{hess_matrix = }")

my_posterior_precision = hess_matrix + torch.eye(hess_size)*1/prior_log_sigma.exp()
my_covariance_matrix = torch.inverse(my_posterior_precision)
print(f"{hess_matrix + torch.eye(hess_size)*1/prior_log_sigma.exp()}")
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







# Totolist. 
# 1) invoke manifold class
# 2) set model to the MAP
# 3) sample from LA approximation
# 4) calc geodesic



class Manifold():
    def __init__(self, model, task_type, device="cpu", prior_sigma=0.1, target_sigma=0.1):
        super(Manifold, self).__init__()
        self.model = model.to(device)
        self.task_type = task_type
        self.device = device

        if self.task_type == "regression":
            self.criterion = torch.nn.MSELoss()
        elif self.task_type == "classification":
            self.criterion = torch.nn.CrossEntropyLoss()
        else:
            raise ValueError("Likelihood not recognized")
        self.prior_sigma = prior_sigma if type(prior_sigma) == torch.Tensor else torch.tensor(prior_sigma)
        self.target_sigma = target_sigma if type(target_sigma) == torch.Tensor else torch.tensor(target_sigma)

        self.MAP = None
        self.MAP_covariance = None
        self.LA = Laplace(model, likelihood=self.task_type, hessian_structure="full", subset_of_weights='all')
        
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
        self.register_this_as_map()

    def register_this_as_map(self):
        self.MAP = torch.nn.utils.parameters_to_vector(model.parameters()).clone().detach()
        self.LA.fit(self.train_loader)
        self.MAP_covariance = self.LA.posterior_covariance.clone().detach()

    def set_to_map(self):
        if self.MAP is not None:
            torch.nn.utils.vector_to_parameters(self.MAP, self.model.parameters())
        else: raise ValueError("No MAP found")

    def set_weights(self, theta):
        torch.nn.utils.vector_to_parameters(theta, self.model.parameters())

    def loss_bck(self):
        return criterion(self.model(self.x_train), self.y_train)
    
    def mse_loss(self, pred=None, target=None):
        if target is None:
            target = self.y_train
        #print(f"{pred.shape = }")
        #print(f"{target.shape = }")
        return (1.0 / self.noise) * self.criterion(pred, target)

    def loss(self, pred=None, target=None):
        #print(f"{x.shape = }")
        #print(f"{y.shape = }")
        if target is None:
            target = self.y_train
        param_norm = torch.nn.utils.parameters_to_vector(self.model.parameters()).norm()**2 / (2 * self.prior_sigma**2)
        #print(f"{param_norm = }")
        #print(f"{pred.shape = }")
        #print(f"{target.shape = }")
        #print(f"{N = }")
        #print(f"{self.target_sigma = }")

        loss_from_pred = self.criterion(pred, target)*x_train.shape[0]/pred.shape[0] / (2 * self.target_sigma**2)
        loss = loss_from_pred #+ param_norm
        return loss #self.mse_loss(x, y) + self.regularization * torch.nn.utils.parameters_to_vector(model.parameters()).norm()**2

    def gradient_bck(self):
        #params = torch.nn.utils.parameters_to_vector(self.model.parameters())
        grads = torch.autograd.grad(self.loss(), self.model.parameters())
        return torch.cat([parm.flatten() for parm in list(grads)])   
    
    def gradient(self):
        #params = torch.nn.utils.parameters_to_vector(self.model.parameters())
        pred = self.model(self.x_train)
        grads = torch.autograd.grad(self.loss(pred, self.y_train), self.model.parameters())  # D times N
        grads_flat = torch.cat([parm.flatten() for parm in list(grads)])
        return grads_flat

    def covariance(self):
        self.LA.fit(self.train_loader)
        return self.LA.posterior_covariance
    
    def hess(self):
        self.LA.fit(self.train_loader)
        return la.posterior_precision

    def posterior_sample(self, n_samples=1):
        samples = torch.distributions.MultivariateNormal(self.MAP, self.MAP_covariance).sample((n_samples,))
        return samples
        #return self.LA.sample(n_samples=n_samples)

    def velocity_sample(self, n_samples=1):
        samples = torch.distributions.MultivariateNormal(torch.zeros_like(self.MAP), self.MAP_covariance).sample((n_samples,))
        return samples
        #return self.LA.sample(n_samples=n_samples)


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

        acc = -(grad_val * (1 / (1 + grad_val.norm()**2)) * (v.T @ hess_val @ v)).flatten()
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

[p for p in model.parameters()]
print([p for p in model.parameters()])
mymanifold = Manifold(model, task_type="regression", prior_sigma=prior_log_sigma.exp(), target_sigma=target_log_sigma.exp())
print(f"{mymanifold.MAP = }")
mymanifold.set_train_data(x_train, y_train)
mymanifold.register_this_as_map()
# mymanifold.fit(epochs=5000, lr=0.1, verbose=True)
mymanifold.gradient()

ts = np.linspace(0, 1, 100)

sample = mymanifold.velocity_sample(n_samples=1)[0]
sol = mymanifold.expmap(mymanifold.MAP, sample)

fig, ax = plt.subplots() 
ax.plot(sol.sol(ts)[0], sol.sol(ts)[1])
ax.scatter(x=posterior_samples[:,0], y=posterior_samples[:,1], c="red")
ax.scatter(x=mymanifold.MAP[0], y=mymanifold.MAP[1], c="blue")
ax.scatter(x=sample[0] + mymanifold.MAP[0], y=sample[1] +mymanifold.MAP[1], c="green")
# plot the laplace approximation
stds = torch.tensor([1, 2, 3, 4])
cov = la.posterior_covariance; mean = la.mean
eigvals, eigvecs = torch.linalg.eigh(cov)
angle = torch.rad2deg(torch.atan2(eigvecs[1, 0], eigvecs[0, 0]))
width, height = torch.sqrt(eigvals).unsqueeze(1) * 2 * stds.unsqueeze(0)
for curveno in range(stds.shape[0]):
    ellipse = plt.matplotlib.patches.Ellipse(mean, width[curveno], height[curveno], angle=angle, fill=False, edgecolor="blue", linewidth=2)
    ax.add_patch(ellipse)
ax.set_xlim(-.1, 1.5)
ax.set_ylim(-2, 2)