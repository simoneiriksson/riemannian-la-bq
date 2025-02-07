import os
import sys
# set working directory
os.chdir("../riemannian_la")
print(os.getcwd())

import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

from models import LinearModel, FunctionApproximatorModel
from train import train
from laplace_approx import Laplace
from MCMC_sampler import MCMC_sampler
from getdata import make_loaders, torch_seed, gen_model_data, gen_log_regression_data
from utils import NegLogLik_regression
from integration import integrator
from utils import loss_func_from_target_sigma, make_functional_fwd_xs, functional_loss, functional_loss_for_vmap, sum_loss, neglog_loss, tensify
from discrete_sampler import discrete_function_sampler, discrete_model_sampler
from riemann_sampler import Riemann_sampler, riemann_plotter
from BQ_rays_subspace import BayesianQuadrature_rays
from emukit.quadrature.methods import VanillaBayesianQuadrature, BoundedBayesianQuadrature
from emukit.model_wrappers import GPyModelWrapper


num_features=1
model = LinearModel(num_features=num_features, num_outputs=1, bias = True)

# setting all parameters to 1
init_params = torch.nn.utils.parameters_to_vector(model.parameters())
torch.nn.utils.vector_to_parameters(torch.ones_like(init_params), model.parameters())
init_params = torch.nn.utils.parameters_to_vector(model.parameters())

prior_sigma = 2.
target_sigma = 1.

start_x = -2.5
end_x = 2.5
#input_dist = lambda x: torch.rand(x, num_features)*(end_x-start_x) + start_x
def input_dist(N):
    xs = torch.rand(N*10, num_features)*(end_x-start_x) + start_x
    index = (1-((xs > -.5) & (xs < 0.5)).int()).nonzero()[:,0]
    return xs[index][:N]

train_loader, test_loader = gen_model_data(model, input_dist, num_train_samples=10, 
                                           num_test_samples=10, noise_std=target_sigma, seed=2, batch_size=1)

pre_trained_params = torch.nn.utils.parameters_to_vector(model.parameters())
xs, ys = train_loader.dataset.dataset.tensors
xs_plt = torch.linspace(start_x, end_x, 100).unsqueeze(1)
plt.scatter(xs[:, 0].detach(), ys.detach(), c="b", label="data")
ys_plt_pretrained = model(xs_plt).detach()
plt.plot(xs_plt[:, 0].detach(), ys_plt_pretrained, c="r", label="true function")
plt.show()

# # Analytical solution for linear model:
Phi = torch.cat([xs, torch.ones(xs.shape[0],1)], dim=1)
prior_Sigma = torch.eye(2) *prior_sigma**2
prior_mu = torch.zeros(2)

S_inverse = prior_Sigma.inverse() + target_sigma**-2 * Phi.T @ Phi
mu = S_inverse.inverse() @ (prior_Sigma.inverse() @ prior_mu + target_sigma**-2 * (Phi.T @ ys.squeeze()))
print(f"Analytical solution:")
print(f"{mu=}")
print(f"{S_inverse=}")
print(f"Covariance: {S_inverse.inverse()}")
Phi_plt = torch.cat([xs_plt, torch.ones(xs_plt.shape[0],1)], dim=1)
mean = Phi_plt @ mu
Phi_plt.shape
epistemic_var = (Phi_plt @ S_inverse.inverse() * Phi_plt).sum(dim=1)
aleatoric_var = target_sigma**2
#std[0]
#std = torch.sqrt(torch.stack([target_sigma + Phi_plt_ @ S_inverse.inverse() @ Phi_plt_ for Phi_plt_ in Phi_plt]))
#std.shape

plt.scatter(xs[:, 0].detach(), ys.detach())
plt.plot(xs_plt[:, 0].detach(), mean)
plt.fill_between(xs_plt[:, 0].detach(), mean-epistemic_var.sqrt(), mean+epistemic_var.sqrt(), alpha=0.3)
plt.fill_between(xs_plt[:, 0].detach(), mean-(aleatoric_var + epistemic_var).sqrt(), mean+(aleatoric_var + epistemic_var).sqrt(), alpha=0.3)
plt.show()


# We now have the analytical posterior and can set the model parameters to the analytical posterior
torch.nn.utils.vector_to_parameters(mu, model.parameters())
ys_plt_trained = model(xs_plt).detach()
####################################
# now we do it with discrete integration
####################################
loss_fn = NegLogLik_regression(target_sigma=target_sigma)

limits_t = (mu + 2 * torch.stack([-S_inverse.inverse().diag().sqrt() , S_inverse.inverse().diag().sqrt()])).T

# loop through 2d array limits_1 and make list of lists:
limits = [[l2.item() for l2 in l1] for l1 in limits_t]
n_mesh = 100
discrete_sampler = discrete_model_sampler(model, loss_fn=loss_fn, xs=xs, ys=ys, limits=limits, n_mesh=n_mesh, normalize_weights=True, prior_sigma=prior_sigma)
posterior_samples, weights = discrete_sampler.samples_and_weights()
weights.sum()

parametersubset = dict(model.named_parameters())
integral, function_values, weights, posterior_samples = integrator(discrete_sampler, make_functional_fwd_xs(model), parametersubset, xs_plt)

mean = ((weights.reshape(-1, 1) * function_values[:,:, 0]).pow(1)).sum(dim=0).detach()

epistemic_var = (((function_values[:,:, 0] - mean.unsqueeze(0)).pow(2))*weights.unsqueeze(1)).sum(dim=0).detach()
second_central_moment = (weights.reshape(-1, 1) * (function_values[:,:, 0]).pow(2)).sum(dim=0).detach()
epistemic_var = second_central_moment - mean.pow(2)

aleatoric_var = target_sigma**2
plt.scatter(xs[:, 0].detach(), ys.detach())
plt.plot(xs_plt[:, 0].detach(), mean)
plt.fill_between(xs_plt[:, 0].detach(), mean-epistemic_var.sqrt(), mean+epistemic_var.sqrt(), alpha=0.3)
plt.fill_between(xs_plt[:, 0].detach(), mean-(aleatoric_var + epistemic_var).sqrt(), mean+(aleatoric_var + epistemic_var).sqrt(), alpha=0.3)
plt.show()


####################################
# Ok, we have trained the model. Now we can try to fit a Laplace approximation to the posterior
####################################
loss_fn = NegLogLik_regression(target_sigma=target_sigma)
laplace = Laplace(model, xs=xs, ys=ys, prior_sigma=prior_sigma, loss_fn=loss_fn, n_posterior_samples=100)

_=laplace.fit_subspace(fitting_type="GGN", xs=xs, ys=ys, subspace_rank=0)
#_=laplace.fit(fitting_type="GGN", xs=xs, ys=ys)
print(f"Covariance: {laplace.covariance}")
print(f"Mean: {laplace.mean}")

laplace.make_posterior_sample(n_samples=10)

integral_la, function_values_lp, weights_lp, posterior_samples_la = integrator(sampler=laplace, model_func=make_functional_fwd_xs(model), xs=xs_plt)

mean = integral_la.detach()[:, 0]
epistemic_var = (function_values_lp[:,:, 0]-mean.unsqueeze(0)).pow(2).mean(dim=0).detach()
aleatoric_var = target_sigma**2
plt.scatter(xs[:, 0].detach(), ys.detach())
plt.plot(xs_plt[:, 0].detach(), mean)
plt.fill_between(xs_plt[:, 0].detach(), mean-epistemic_var.sqrt(), mean+epistemic_var.sqrt(), alpha=0.3)
plt.fill_between(xs_plt[:, 0].detach(), mean-(aleatoric_var + epistemic_var).sqrt(), mean+(aleatoric_var + epistemic_var).sqrt(), alpha=0.3)
plt.show()

####################################
# Laplace subspace integration
####################################
loss_fn = NegLogLik_regression(target_sigma=target_sigma)
laplace = Laplace(model, xs=xs, ys=ys, prior_sigma=prior_sigma, loss_fn=loss_fn, n_posterior_samples=100, subspace_rank=1)

_=laplace.fit(fitting_type="hessian", xs=xs, ys=ys)
#print(f"Covariance: {laplace.covariance}")
#print(f"Mean: {laplace.mean}")

laplace.make_posterior_sample(n_samples=10)

integral_la, function_values_lp, weights_lp, posterior_samples_la = integrator(sampler=laplace, model_func=make_functional_fwd_xs(model), xs=xs_plt)

mean = integral_la.detach()[:, 0]
epistemic_var = (function_values_lp[:,:, 0]-mean.unsqueeze(0)).pow(2).mean(dim=0).detach()
aleatoric_var = target_sigma**2
plt.scatter(xs[:, 0].detach(), ys.detach())
plt.plot(xs_plt[:, 0].detach(), mean)
plt.fill_between(xs_plt[:, 0].detach(), mean-epistemic_var.sqrt(), mean+epistemic_var.sqrt(), alpha=0.3)
plt.fill_between(xs_plt[:, 0].detach(), mean-(aleatoric_var + epistemic_var).sqrt(), mean+(aleatoric_var + epistemic_var).sqrt(), alpha=0.3)
plt.show()

####################################
# Riemann integration
####################################
R_sampler = Riemann_sampler(model, xs=xs, ys=ys, loss_fn=loss_fn, prior_sigma=prior_sigma,
                            n_posterior_samples=10, subspace_rank=2)
_=R_sampler.fit(fitting_type="GGN")
R_sampler.covariance.shape

_=R_sampler.make_posterior_sample(n_samples=10)
#fig, ax = riemann_plotter(R_sampler, plot_traject=False, sample_markers=None)
#ax.scatter(R_sampler.posterior_samples_la[:, 0].detach(), R_sampler.posterior_samples_la[:, 1].detach(), color="k")
#plt.show()
integral, function_values, weights, posterior_samples = integrator(R_sampler, model_func=make_functional_fwd_xs(model), xs=xs_plt)

mean = integral.detach()[:, 0]
epistemic_var = (function_values[:,:, 0]-mean.unsqueeze(0)).pow(2).mean(dim=0).detach()
aleatoric_var = target_sigma**2
plt.scatter(xs[:, 0].detach(), ys.detach())
plt.plot(xs_plt[:, 0].detach(), mean)
plt.fill_between(xs_plt[:, 0].detach(), mean-epistemic_var.sqrt(), mean+epistemic_var.sqrt(), alpha=0.3)
plt.fill_between(xs_plt[:, 0].detach(), mean-(aleatoric_var + epistemic_var).sqrt(), mean+(aleatoric_var + epistemic_var).sqrt(), alpha=0.3)
plt.show()



####################################
# Riemannian subspace integration
####################################
R_sampler = Riemann_sampler(model, xs=xs, ys=ys, loss_fn=loss_fn, prior_sigma=prior_sigma,  
                            n_posterior_samples=10, subspace_rank=1)
R_sampler.fit(fitting_type="GGN")
R_sampler.make_posterior_sample(n_samples=10)
#riemann_plotter(R_sampler)

integral, function_values, weights, posterior_samples = integrator(R_sampler, model_func=make_functional_fwd_xs(model), xs=xs_plt)

mean = integral.detach()[:, 0]
epistemic_var = (function_values[:,:, 0]-mean.unsqueeze(0)).pow(2).mean(dim=0).detach()
aleatoric_var = target_sigma**2
plt.scatter(xs[:, 0].detach(), ys.detach())
plt.plot(xs_plt[:, 0].detach(), mean)
plt.fill_between(xs_plt[:, 0].detach(), mean-epistemic_var.sqrt(), mean+epistemic_var.sqrt(), alpha=0.3)
plt.fill_between(xs_plt[:, 0].detach(), mean-(aleatoric_var + epistemic_var).sqrt(), mean+(aleatoric_var + epistemic_var).sqrt(), alpha=0.3)
plt.show()


####################################
# Bayesian Quadrature integration
####################################

R_sampler = Riemann_sampler(model, xs=xs, ys=ys, loss_fn=loss_fn, prior_sigma=prior_sigma,  subspace_rank=2)
_=R_sampler.fit(fitting_type="GGN")


"gaussian_rescaled"
"lebesgue_rescaled"
"lebesgue"

BQ = BayesianQuadrature_rays(R_sampler, evaluation_model=model, measure="gaussian_rescaled", integral_bounds_std=4, 
                             GP_lengthscale=1.0, GP_variance=1.0, num_timesteps=7, use_ray_acqusition=True, 
                             use_rays=True, 
                             theta_space_plot_limits=[[-1,1], [-1,1]], xs = xs[0], parametersubset=None)

for i in range(8):
    integral_mean, integral_variance = BQ.step()
    print(f"{i = }, {integral_mean = }, {integral_variance = }")
    # if i%1 ==0:
fig, axes = BQ.plot()
BQ.thetas
means, epistemic_var, vars = BQ.pred_BQ(xs_plt, get_measure_variance=True, transform_type="sqrt")
aleatoric_var = target_sigma**2
import numpy as np
#plt.scatter(xs_plt.unsqueeze(1).repeat([1, preds.shape[1], 1]), preds, marker="x", c="y", label="second central moment")
#plt.scatter(xs_plt, second_central_moments_mus - first_central_moments_mus**2, marker="x", c="g", label="first central moment")
plt.plot(xs_plt, means, c="g", label="mean")
plt.fill_between(xs_plt[:,0], means - np.sqrt(aleatoric_var + epistemic_var), means + np.sqrt(aleatoric_var + epistemic_var), alpha=0.3, color="darkgreen", label="epistemic + aleatoric")
plt.fill_between(xs_plt[:,0], means - np.sqrt(epistemic_var), means + np.sqrt(epistemic_var), alpha=0.3, color="blue", label="epistemic")
plt.plot(xs_plt[:, 0].detach(), ys_plt_trained, c="r", label="true function")
plt.scatter(xs[:, 0].detach(), ys.detach(), c="b", marker=".", label="data")
plt.legend()

