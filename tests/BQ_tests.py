import os
import sys
# set working directory
print("working dir:", os.getcwd())
os.chdir("../riemannian_la")

import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.distributions.multivariate_normal import _precision_to_scale_tril
from utils import tensify, loss_func_from_target_sigma, make_functional_fwd_xs, vector_to_parameterdict, make_functional_fwd
from GGN_hessian import GGN_hessian_from_loader
from hessian import hessian_from_model_loss_and_data, hessian_dict_to_matrix, hessian_from_loader, hessian_from_func
from riemannian_la.utils import NegLogLik_regression, NegLogLik_classification, iid_gaussian_prior
from torch.func import grad, jvp, vjp, hessian, jacfwd, jacrev, vmap, functional_call
from laplace_approx import Laplace
from scipy.integrate import solve_ivp
from utils import make_functional_fwd_xs, functional_loss_for_vmap, neglog_loss, make_functional_fwd_vector
from models import functional_banana, Model_from_func, functional_d1_halfcircle, functional_d1_2, functional_d1, functional_d1_fourth_degree_poly, LinearModel, functional_d1_normal
from matplotlib import pyplot as plt
from MCMC_sampler import MCMC_sampler
import torchdiffeq
import seaborn as sns
import pandas as pd
from riemann_sampler import Riemann_sampler, riemann_plotter

import GPy
from FullGaussianMeasure import FullGaussianMeasure
from emukit.model_wrappers.gpy_quadrature_wrappers import BaseGaussianProcessGPy, RBFGPy
from emukit.quadrature.kernels import QuadratureRBFLebesgueMeasure, QuadratureRBFGaussianMeasure
from emukit.quadrature.measures import LebesgueMeasure, GaussianMeasure

from emukit.quadrature.methods import VanillaBayesianQuadrature
from emukit.quadrature.acquisitions import IntegralVarianceReduction
from emukit.core.optimization import GradientAcquisitionOptimizer
from emukit.core.parameter_space import ParameterSpace
import numpy as np

from RayAcquisition import RayAcquisition

#################
# Now for 1d test

a = 1
d1_function = functional_d1_halfcircle(a)
#d1_function = functional_d1_normal(0,1)

displace = 0
d1_function_moved = lambda x: d1_function(x+displace)
xs = torch.tensor([-1.0]).unsqueeze(0)*displace
ys = d1_function_moved(xs[0]).unsqueeze(0)

# xs_plt = np.linspace(-2, 2, 100)
# ys_plt = d1_function_moved(torch.tensor(xs_plt)).detach().numpy()
# plt.plot(xs_plt, ys_plt)

class tiny_ridiculess_model_class(torch.nn.Module):
    def __init__(self, n_params):
        super().__init__()
        self.params = torch.nn.Parameter(torch.arange(n_params).float())
    def forward(self, x):
        #return torch.sin(self.params*3.1).sum().repeat(x.shape[0], 1)**2
        #return torch.ones(x.shape[0], 1)
        return (self.params+displace).sum().repeat(x.shape[0], 1)**2
        #return self.params.sum().repeat(x.shape[0], 1)
        
evaluation_model = tiny_ridiculess_model_class(n_params=1)
functional_evaluation_model = make_functional_fwd_xs(evaluation_model)  # make functional version of model

d1_model = Model_from_func(d1_function_moved, input_shape=[1])
torch.nn.utils.vector_to_parameters(xs[0], d1_model.parameters())
parametersubset = dict(d1_model.named_parameters())

R_sampler = Riemann_sampler(d1_model, xs=xs, ys=ys, loss_fn=neglog_loss(), prior_sigma=0, subspace_rank=1)
R_sampler.fit(fitting_type="hessian")

torch.manual_seed(0)
"gaussian_rescaled"
"lebesgue_rescaled"
"lebesgue"

BQ = BayesianQuadrature_rays(R_sampler, evaluation_model, measure="lebesgue_rescaled", integral_bounds_std=4, 
                             GP_lengthscale=1.0, GP_variance=1.0, num_timesteps=10, use_ray_acqusition=True, use_rays=True, theta_space_plot_limits=[-1,1])

for i in range(4):
    integral_mean, integral_variance = BQ.step()
    print(f"{i = }, {integral_mean = }, {integral_variance = }")
    # if i%1 ==0:
fig, axes = BQ.plot()
plt.show()

function_vals = torch.stack([BQ.functional_fwd(obs) for obs in torch.tensor(BQ.thetas)])[:,0]
plt_model_loss = torch.stack([BQ.Rsampler.f_loss(obs) for obs in torch.tensor(BQ.thetas)])

meh = (function_vals * (-plt_model_loss).exp())
plt.scatter(BQ.thetas, BQ.integrand_values, c="r")
plt.scatter(BQ.thetas, meh, c="b")


#################
# Now for 2d test
curvature = 0.1
banana_function = functional_banana(curvature=curvature, sigma_x=2.0, sigma_y=1.0)

displace = 1.0
banana_function_moved = lambda x: banana_function(x+displace)
xs = torch.tensor([-1.0, -1.0]).unsqueeze(0)*displace
ys = banana_function_moved(xs[0]).unsqueeze(0)

class tiny_ridiculess_model_class(torch.nn.Module):
    def __init__(self, n_params):
        super().__init__()
        self.params = torch.nn.Parameter(torch.arange(n_params).float())
    def forward(self, x):
        #return torch.sin(self.params*3.1).sum().repeat(x.shape[0], 1)**2
        #return torch.ones(x.shape[0], 1)
        return (self.params+displace).sum().repeat(x.shape[0], 1)**2
        #return self.params.sum().repeat(x.shape[0], 1)+2
        
evaluation_model = tiny_ridiculess_model_class(n_params=2)
functional_evaluation_model = make_functional_fwd_xs(evaluation_model)  # make functional version of model

banana_model = Model_from_func(banana_function_moved, input_shape=[2])
torch.nn.utils.vector_to_parameters(xs[0], banana_model.parameters())
parametersubset = dict(banana_model.named_parameters())

R_sampler = Riemann_sampler(banana_model, xs=xs, ys=ys, loss_fn=neglog_loss(), prior_sigma=0, subspace_rank=None)
R_sampler.fit(fitting_type="hessian")

evaluation_model = tiny_ridiculess_model_class(n_params=2)

torch.manual_seed(0)
BQ = BayesianQuadrature_rays(R_sampler, evaluation_model, measure="gaussian_rescaled", integral_bounds_std=4, 
                             GP_lengthscale=1.0, GP_variance=1.0, num_timesteps=4, use_ray_acqusition=True, use_rays=True)

for i in range(16):
    integral_mean, integral_variance = BQ.step()
    print(f"{i = }, {integral_mean = }, {integral_variance = }")
    # if i%1 ==0:
fig, axes = BQ.plot()
plt.show()
