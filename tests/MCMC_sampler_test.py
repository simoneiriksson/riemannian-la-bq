import os
import sys
# set working directory
os.chdir("../riemannian_la")
# print(os.getcwd())
from models import LinearModel, Model_from_func
from getdata import gen_log_regression_data
from train import train
from laplace_approx import Laplace, vector_to_parameterdict
from utils import make_functional_fwd, loss_func_from_target_sigma, make_functional_fwd_xs, functional_loss, functional_loss_for_vmap, sum_loss, neglog_loss
from discrete_sampler import discrete_function_sampler, discrete_model_sampler
from integration import integrator
from torch.func import grad, jvp, vjp, hessian, jacfwd, jacrev, vmap, functional_call
from matplotlib import pyplot as plt
from laplace_approx import Laplace
import torch
from models import functional_banana
from MCMC_sampler import MCMC_sampler

import hamiltorch

# First, lets sample from the banana function using HMC, to check that it works!
hamiltorch.set_random_seed(123)
params_init = torch.zeros(2)
N=1000
step_size=.3
L=5
banana_function = functional_banana(curvature=1.0, sigma_x=2.0, sigma_y=1.0)
log_prob_banana = lambda params: banana_function(params).log()
params_hmc = hamiltorch.sample(log_prob_func=log_prob_banana, params_init=params_init, num_samples=N,
                               step_size=step_size, num_steps_per_sample=L)

tens_params_hmc = torch.stack(params_hmc)
plt.scatter(tens_params_hmc[:, 0], tens_params_hmc[:, 1])

# this plot looks good!

# Now, let us make a model from the banana function, which has the x/y coordinates as parameters
# and the banana function value as output
# With this, we will build a another function that takes a parameter vector and an input tensor, and returns the model output tensor
xs = torch.tensor([0.0, 0.0]).unsqueeze(0)
ys = banana_function(xs[0]).unsqueeze(0)

banana_function = functional_banana(curvature=1.0, sigma_x=2.0, sigma_y=1.0)
log_prob_banana = lambda params: banana_function(params).log()
log_banana_model = Model_from_func(log_prob_banana, input_shape=[2])

parametersubset = dict(log_banana_model.named_parameters())
f_banana_model = make_functional_fwd_xs(log_banana_model)
loss_fn = sum_loss()
log_prob_func = functional_loss_for_vmap(f_banana_model, parametersubset, loss_fn, xs, ys)

params_hmc = hamiltorch.sample(log_prob_func=log_prob_func, params_init=params_init, num_samples=N,
                               step_size=step_size, num_steps_per_sample=L)

tens_params_hmc = torch.stack(params_hmc)
plt.scatter(tens_params_hmc[:, 0], tens_params_hmc[:, 1])


# Now, let us make a model from the banana function, which has the x/y coordinates as input
# and the banana function value as output
# With this, we will build a function that takes a parameter vector and an input tensor, and returns the model output tensor

banana_function = functional_banana(curvature=1.0, sigma_x=2.0, sigma_y=1.0)
banana_model = Model_from_func(banana_function, input_shape=[2])

xs = torch.tensor([0.0, 0.0]).unsqueeze(0)
ys = banana_function(xs[0]).unsqueeze(0)
params_init = torch.zeros(2)


parametersubset = dict(banana_model.named_parameters())
f_banana_model = make_functional_fwd_xs(banana_model)

loss_fn = lambda preds, target: torch.sum(preds.log())

log_prob_func = functional_loss_for_vmap(f_banana_model, parametersubset, loss_fn, xs, ys)

params_hmc = hamiltorch.sample(log_prob_func=log_prob_func, params_init=params_init, num_samples=N,
                               step_size=step_size, num_steps_per_sample=L)

tens_params_hmc = torch.stack(params_hmc)
plt.scatter(tens_params_hmc[:, 0], tens_params_hmc[:, 1])
# This also works!

# Now we use the MCMC_sampler class to do the same thing
banana_function = functional_banana(curvature=1.0, sigma_x=2.0, sigma_y=1.0)
banana_model = Model_from_func(banana_function, input_shape=[2])
loss_fn = lambda preds, target: torch.sum(preds.log())

const_prior = lambda params: torch.tensor(0.0)

xs = torch.tensor([0.0, 0.0]).unsqueeze(0)
ys = banana_function(xs[0]).unsqueeze(0)
params_init = torch.zeros(2)

parametersubset = dict(banana_model.named_parameters())

sampler = MCMC_sampler(banana_model, parametersubset, xs=xs, ys=ys, loss_fn=loss_fn, prior_logprob=const_prior)
tens_params_hmc = sampler.make_posterior_sample(1000)
plt.plot(tens_params_hmc[:, 0], tens_params_hmc[:, 1])
# This also also works!



