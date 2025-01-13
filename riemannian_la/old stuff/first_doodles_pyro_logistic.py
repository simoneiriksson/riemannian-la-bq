import torch
import numpy as np
import matplotlib.pyplot as plt
import math
from torch.utils.data import DataLoader, TensorDataset
from torch_to_pyro import pyro_model_from_pytorch

import pyro
import pyro.distributions as dist
from pyro.contrib.autoguide import AutoDiagonalNormal, AutoMultivariateNormal
from pyro.infer import MCMC, NUTS, HMC, SVI, Trace_ELBO
from pyro.optim import Adam, ClippedAdam
import seaborn as sns
import itertools
from pyro.infer import Predictive
from pyro.infer.autoguide.guides import AutoLaplaceApproximation
from pyro.ops.hessian import hessian
from plotting import distributions_plot

def fn(xs):
    return torch.column_stack([xs[:,0], xs[:,1], xs[:,0]**2 + xs[:,1]**2])

def fn(xs):
    return torch.column_stack([xs[:,0]**1, xs[:,0]**2, xs[:,0]**3])

def generate_data(xs, fn, weights, sigma_y=0.1):
    ys = fn(xs) @ weights + torch.normal(torch.tensor(0.0), torch.tensor(1.)*sigma_y, size=(xs.shape[0],1))
    return ys

def generate_data(N, mu=None, Sigma=None, class_weights=None, sigma_y=0.1):

    if (mu is None) or (Sigma is None) or (class_weights is None):
        ndim = 2

        class_weights=torch.ones(ndim)*1.0/ndim
        #A = torch.row_stack([torch.eye(ndim).unsqueeze(0), (torch.ones(ndim)- torch.eye(ndim)).unsqueeze(0)])
        Sigma = torch.tensor([[[1.0, 0.0], [0.0, 1.0]], [[1.0, -1.0], [1.0, 1.0]]])*0.5
        mu = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    ndim = mu.shape[0]
    eye = torch.eye(ndim)
    null = torch.zeros(ndim)

    xs_std = torch.distributions.MultivariateNormal(null,eye).sample((100,))
    classes = torch.distributions.Categorical(class_weights).sample((100,))
    xs = torch.bmm(Sigma[classes], xs_std.unsqueeze(-1)).squeeze(-1) + mu[classes] 
    return xs, classes


class MyLogregModel(torch.nn.Module):
    def __init__(self, M, K): # M is the number of features, K is the number of classes
        super(MyLogregModel, self).__init__()
        self.M = M
        self.K = K
        self.weights = torch.nn.Parameter(torch.rand((M, K)))
        
    def forward(self, xs):
        return xs @ self.weights

N=100
M=2
K = 2
ndim = K
class_weights=torch.ones(ndim)*1.0/ndim
Sigma = torch.tensor([[[1.0, 0.0], [0.0, 1.0]], [[1.0, -1.0], [1.0, 1.0]]])*0.5
mu = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
xs, ys = generate_data(N=N, mu=mu, Sigma=Sigma, class_weights=class_weights)

ys.shape
xs.shape

x_test = xs[0:N//5, :]
y_test = ys[0:N//5]
x_train = xs[N//5:, :]
y_train = ys[N//5:]
x_train.shape, y_train.shape, x_test.shape, y_test.shape

device="cpu"

def pyro_model(x, y=None):
    weights = pyro.sample("weights", dist.Normal(torch.zeros((M, K), device=device), torch.ones((M, K), device=device)).to_event(1))
    bias = pyro.sample("bias", dist.Normal(torch.zeros((K), device=device), torch.ones((K), device=device)).to_event(0))
    class_probs = pyro.deterministic("class_probs", x @ weights + bias)
    print(f"{class_probs.shape=}")
    with pyro.plate('data', x.shape[0]):
        return pyro.sample('obs', dist.Categorical(logits=class_probs), obs=y)

##########################################

# Laplace approximation
def get_laplace(pyro_model, x_train, y_train, num_iters=400, lr=0.01):
    #num_iters=400; lr=0.01
    delta_guide = AutoLaplaceApproximation(pyro_model)
    svi = SVI(pyro_model, delta_guide, Adam({"lr": lr}), loss=Trace_ELBO())
    
    pyro.clear_param_store()
    #x_train.shape, y_train[:,0].shape
    for i in range(num_iters):
        loss = svi.step(x_train.to(device), y_train.to(device))
        if i % 50 == 0:
            print(f"i = {i}, loss = {loss}")
    #svi.run(x_train.to(device), y_train[:,0].to(device), num_iters)
    guide_trace = pyro.poutine.trace(delta_guide).get_trace(x_train)
    model_trace = pyro.poutine.trace(pyro.poutine.replay(delta_guide.model, trace=guide_trace)).get_trace(x_train)
    loss = guide_trace.log_prob_sum() - model_trace.log_prob_sum()
    H = hessian(loss, delta_guide.loc)
    loc = delta_guide.loc.detach()
    cov = H.inverse()
    model_guide_parameters = {}
    for k in model_trace.nodes.keys():
        #print(f"{k=}")
        if model_trace.nodes[k]["type"] == "sample":
            #print(f"{model_trace.nodes[k]['value'].shape=}")
            model_guide_parameters[k] = model_trace.nodes[k]["value"].shape
    return loc, cov, H, delta_guide, model_guide_parameters

LA_loc, LA_cov, LA_H, delta_guide, model_guide_parameters = get_laplace(pyro_model, x_train, y_train, num_iters=1000, lr=0.005)

eigenvals, eigenvectors = torch.linalg.eigh(LA_cov)

subspace_dim = 2
subspace = eigenvectors[:,subspace_dim:]
cov_restricted = subspace @ torch.diag(eigenvals[subspace_dim:]) @ subspace.T
print(f"{cov_restricted=}")

########################################
# MCMC-samples
nuts_kernel = NUTS(pyro_model)
#mcmc = MCMC(nuts_kernel, num_samples=100, warmup_steps=100, num_chains=1)
mcmc = MCMC(nuts_kernel, num_samples=1000, warmup_steps=200, num_chains=1)
mcmc.run(x_train.to(device), y_train[:,0].to(device))

# Show summary of inference results
mcmc.summary()

# Extract samples from posterior
posterior_samples = mcmc.get_samples()
print(f"{posterior_samples['weights'].shape=}")
print(f"{posterior_samples['log_scale'].shape=}")

# plot chains:
fig, axes = plt.subplots(1, 4, figsize=(15, 5))
sns.lineplot(data=posterior_samples["weights"][:,0], ax=axes[0]).set(title="weight 0")
sns.lineplot(data=posterior_samples["weights"][:,1], ax=axes[1]).set(title="weight 1")
sns.lineplot(data=posterior_samples["weights"][:,2], ax=axes[2]).set(title="weight 2")
sns.lineplot(data=posterior_samples["log_scale"][:], ax=axes[3]).set(title="log_scale")
fig.show()

# collect samples in one tensor, in the same order as the laplace approximation
posterior_samples_list = []
posterior_samples_names =[]
for parameter, size in model_guide_parameters.items():
    if parameter in posterior_samples:
        posterior_samples_names += [f"{parameter}_{i}" for i in range(size)]
        posterior_samples_list.append(posterior_samples[parameter].reshape(-1, *size))

posterior_samples_tensor = torch.cat(posterior_samples_list, dim=-1)

print(f"{posterior_samples_tensor.mean(dim=0) = }, \n{posterior_samples_tensor.std(dim=0)**2 = }")
print(f"{LA_loc = }, \n{LA_cov.diag() = }")

print(f"{posterior_samples_tensor.T.cov() = }")
print(f"{LA_cov = }")

fig, axs = distributions_plot(posterior_samples_tensor, LA_loc, LA_cov)
fig.show()


# project MCMC samples onto the subspace spanned by the eigenvectors corresponding to the smallest eigenvalues
# of the laplace transformation.
posterior_samples_tensor_projected = (posterior_samples_tensor - LA_loc) @ subspace
#posterior_samples_tensor_projected.mean(dim=0)
#posterior_samples_tensor_projected.T.cov()

laplace_samples = torch.distributions.MultivariateNormal(LA_loc, LA_cov).sample((1000,))
laplace_samples.T.cov()
posterior_samples_tensor.T.cov()

laplace_samples_projected = (laplace_samples - LA_loc) @ subspace
laplace_samples_projected.T.cov()


plt.scatter(posterior_samples_tensor_projected[:,0], posterior_samples_tensor_projected[:,1], label="MCMC")
plt.scatter(laplace_samples_projected[:,0], laplace_samples_projected[:,1], label="Laplace")
plt.legend()

class projection(torch.nn.Module):
    def __init__(self, subspace, LA_loc):
        super(projection, self).__init__()
        self.subspace = torch.nn.Parameter(subspace)
        self.LA_loc = torch.nn.Parameter(LA_loc)
    
    def project(self, x):
        return (x - self.LA_loc) @ self.subspace @ subspace.T + self.LA_loc

    def represent(self, x):
        return (x - self.LA_loc) @ self.subspace

    def unrepresent(self, x):
        return x @ self.subspace.T + self.LA_loc


