import os
import sys
# set working directory
print(f"{__file__ = }")
print(f"{sys.path = }")
print("working dir:", os.getcwd())
os.chdir("../riemannian_la")
print("working dir:", os.getcwd())
from models import LinearModel, Model_from_func, functional_d1_halfcircle
from getdata import gen_log_regression_data
from train import train
from laplace_approx import Laplace, vector_to_parameterdict
from utils import loss_func_from_target_sigma, make_functional_fwd_xs, functional_loss, functional_loss_for_vmap, sum_loss, neglog_loss, tensify
from discrete_sampler import discrete_function_sampler, discrete_model_sampler
from riemann_sampler import Riemann_sampler, riemann_plotter
from integration import integrator
from torch.func import grad, jvp, vjp, hessian, jacfwd, jacrev, vmap, functional_call
from matplotlib import pyplot as plt
from laplace_approx import Laplace
from MCMC_sampler import MCMC_sampler
import torch
from models import functional_banana
from MCMC_sampler import MCMC_sampler
import seaborn as sns
import pandas as pd

n_mesh = 1000
a = 1.0

#################
# Now for 1d test
func_1d = functional_d1_halfcircle(a=a)
func_1d_model = Model_from_func(func_1d, input_shape=[1])
loss_fn = lambda preds, target: torch.sum()
xs = torch.tensor([0.0]).unsqueeze(0)
ys = func_1d(xs[0]).unsqueeze(0)

const_prior = lambda params: (params<=1.0 and params>=-1.0).float() * torch.tensor(.5)

class tiny_ridiculess_model_class(torch.nn.Module):
    def __init__(self, n_params):
        super().__init__()
        self.params = torch.nn.Parameter(torch.arange(n_params).float())
    def forward(self, x):
        #return torch.sin(self.params*3.1).sum().repeat(x.shape[0], 1)**2
        #return torch.ones(x.shape[0], 1)
        return self.params.sum().repeat(x.shape[0], 1)**2
        #return self.params.sum().repeat(x.shape[0], 1)+2
        
evaluation_model = tiny_ridiculess_model_class(n_params=1)
functional_evaluation_model = make_functional_fwd_xs(evaluation_model)  # make functional version of model

####################################
# 1) discrete integration over function
####################################
#Let's do discrete integration with the banana function
#banana_function = functional_banana(curvature=2.0, sigma_x=2.0, sigma_y=1.0)
#neg_banana = lambda x: -functional_banana(curvature=0.0, sigma_x=1.0, sigma_y=1.0)(x)
span = a*2
limits = [[-span, span]]
discrete_sampler = discrete_function_sampler(func=func_1d, limits=limits, n_mesh=n_mesh, normalize_weights=True)
posterior_samples, weights = discrete_sampler.samples_and_weights()

torch.nn.utils.vector_to_parameters(torch.tensor([0.1]), evaluation_model.parameters())
parametersubset = dict(evaluation_model.named_parameters())
functional_evaluation_model(parametersubset, torch.tensor([0.0]).unsqueeze(0))

integral, function_values, weights, posterior_samples = integrator(discrete_sampler, functional_evaluation_model, parametersubset, xs)
print(f"1) When using discrete integration over 1d FUNCTION we get {integral = }")

print(f"0) Analytical result is integral {1/4}")

####################################
# 3) laplace integration over model
####################################

# and now the laplace approximation with the banana function
func_1d_model = Model_from_func(func_1d, input_shape=[1])
torch.nn.utils.vector_to_parameters(torch.tensor([0.0]), func_1d_model.parameters())
dict(func_1d_model.named_parameters())

laplace = Laplace(func_1d_model, xs=xs, ys=ys, prior_sigma=0, loss_fn=neglog_loss())
laplace.fit(fitting_type="hessian", xs=xs, ys=ys)

for i in range(1, 16):
    n_samples = 2**i
    laplace.make_posterior_sample(n_samples=n_samples)
    integral_la, function_values_lp, weights_lp, posterior_samples_la = integrator(laplace, functional_evaluation_model, parametersubset, xs)
    posterior_samples_la.shape
    print(f"When using the laplace approxiation of posterior of 1d MODEL with {n_samples} we get {integral_la = }")

laplace.make_posterior_sample(n_samples=50)
integral_la, function_values_lp, weights_lp, posterior_samples_la = integrator(laplace, functional_evaluation_model, parametersubset, xs)

#######################################
# 4) MCMC-integration from banana model
#######################################

# and now the laplace approximation with the banana function
func_1d_model = Model_from_func(func_1d, input_shape=[1])
torch.nn.utils.vector_to_parameters(torch.tensor([0.0]), func_1d_model.parameters())
const_prior = lambda params: (params<=1.0 and params>=-1.0).float() * torch.tensor(.5)

loss_fn = lambda preds, target: torch.sum(preds.log())

parametersubset = dict(func_1d_model.named_parameters())
sampler = MCMC_sampler(func_1d_model, parametersubset, xs=xs, ys=ys, loss_fn=loss_fn, prior_logprob=const_prior)
_=sampler.make_posterior_sample(1000)

integral_mcmc, function_values_mcmc, weights_mcmc, posterior_samples_mcmc = integrator(sampler, functional_evaluation_model, parametersubset, xs)

print(f"When using the MCMC sampling of posterior of func_1d MODEL with {n_samples} we get {integral_mcmc = }")



####################################
# 5) Riemannian laplace integration over model in 1d
####################################

func_1d_model = Model_from_func(func_1d, input_shape=[1])
torch.nn.utils.vector_to_parameters(torch.tensor([0.0]), func_1d_model.parameters())
parametersubset = dict(func_1d_model.named_parameters())

R_sampler = Riemann_sampler(func_1d_model, xs=xs, ys=ys, loss_fn=neglog_loss(), prior_sigma=0)
R_sampler.fit(fitting_type="hessian")

_=R_sampler.make_posterior_sample_la(50)
R_params = R_sampler.make_posterior_sample()
integral_la, function_values_lp, weights_lp, posterior_samples_la = integrator(R_sampler, functional_evaluation_model, parametersubset, xs)
print(f"{integral_la = }")

for i in range(1, 10):
    n_samples = 2**i
    _=R_sampler.make_posterior_sample_la(n_samples)
    R_params = R_sampler.make_posterior_sample()
    integral_la, function_values_lp, weights_lp, posterior_samples_la = integrator(R_sampler, functional_evaluation_model, parametersubset, xs)
    print(f"When using the Riemannian laplace approxiation of posterior of func_1d MODEL with {n_samples} we get {integral_la = }")

R_sampler.posterior_samples_la
R_sampler.posterior_samples


