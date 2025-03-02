import os
import sys
# set working directory
print("working dir:", os.getcwd())
os.chdir("../riemannian_la")

import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.distributions.multivariate_normal import _precision_to_scale_tril
from utils import tensify, loss_func_from_target_sigma, make_functional_fwd_xs, vector_to_parameterdict
from GGN_hessian import GGN_hessian_from_loader
from hessian import hessian_from_model_loss_and_data, hessian_dict_to_matrix, hessian_from_loader, hessian_from_func
from riemannian_la.utils import NegLogLik_regression, NegLogLik_classification, iid_gaussian_prior_loss
from torch.func import grad, jvp, vjp, hessian, jacfwd, jacrev, vmap, functional_call
from laplace_approx import Laplace
from scipy.integrate import solve_ivp
from utils import make_functional_fwd_xs, functional_loss_for_vmap, neglog_loss
from models import functional_banana, Model_from_func
from matplotlib import pyplot as plt
from MCMC_sampler import MCMC_sampler
import torchdiffeq
import seaborn as sns
import pandas as pd
from riemann_sampler import Riemann_sampler, riemann_plotter

# Now, let us make a model from the banana function, which has the x/y coordinates as input
# and the banana function value as output
# With this, we will build a function that takes a parameter vector and an input tensor, and returns the model output tensor

banana_function = functional_banana(curvature=1.0, sigma_x=2.0, sigma_y=1.0)
banana_model = Model_from_func(banana_function, input_shape=[2])
torch.nn.utils.vector_to_parameters(torch.tensor([0.0, 0.0]), banana_model.parameters())
const_prior = lambda params: torch.tensor(0.0)
loss_fn = neglog_loss()

parametersubset = dict(banana_model.named_parameters())

xs = torch.tensor([0.0, 0.0]).unsqueeze(0)
ys = banana_function(xs[0]).unsqueeze(0)
params_init = torch.zeros(2)

R_sampler = Riemann_sampler(banana_model, parametersubset, xs=xs, ys=ys, loss_fn=loss_fn, prior_loss=const_prior)
R_sampler.fit(fitting_type="hessian")

_=R_sampler.make_posterior_sample_la(50)
R_params = R_sampler.make_posterior_sample_scipy()


ax, fig = riemann_plotter(R_sampler, sample_markers=".", plot_traject=True, plot_traj_marker=None, max_samples=None, LA_arrows=[1])
plt.show()

# runtime = timeit.timeit(lambda: R_sampler.make_posterior_sample_torchdiffeq(), number=10)
# print(f"{runtime = }")
# R_params_torchdiffeq = R_sampler.make_posterior_sample_torchdiffeq()
# plot_traj(R_sampler)



# lets try the subsapce version
banana_function = functional_banana(curvature=1.0, sigma_x=2.0, sigma_y=1.0)
banana_model = Model_from_func(banana_function, input_shape=[2])
torch.nn.utils.vector_to_parameters(torch.tensor([0.0, 0.0]), banana_model.parameters())
const_prior = lambda params: torch.tensor(0.0)
loss_fn = neglog_loss()

parametersubset = dict(banana_model.named_parameters())

xs = torch.tensor([0.0, 0.0]).unsqueeze(0)
ys = banana_function(xs[0]).unsqueeze(0)
params_init = torch.zeros(2)

R_sampler = Riemann_sampler(banana_model, parametersubset, xs=xs, ys=ys, loss_fn=loss_fn, prior_loss=const_prior, subspace_rank=1)
R_sampler.fit(fitting_type="hessian")

_=R_sampler.make_posterior_sample_la(50)
R_params = R_sampler.make_posterior_sample_scipy()


ax, fig = riemann_plotter(R_sampler, sample_markers=".", plot_traject=True, plot_traj_marker=None, max_samples=None, LA_arrows=[1])
plt.show()