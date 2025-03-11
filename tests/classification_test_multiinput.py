import os
import sys
# set working directory
os.chdir("../riemannian_la")
print(os.getcwd())

import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import matplotlib 
from models import LinearModel, FunctionApproximatorModel, SineModel, SineModel_2d
from train import train
from laplace_approx import Laplace
from MCMC_sampler import MCMC_sampler
from getdata import make_loaders, torch_seed, gen_model_data, gen_log_regression_data
from utils import NegLogLik_regression, NegLogLik_classification, identity_func
from integration import integrator
from utils import loss_func_from_target_sigma, make_functional_fwd_xs, functional_loss, functional_loss_for_vmap, sum_loss, neglog_loss, tensify
from discrete_sampler import discrete_function_sampler, discrete_model_sampler
from riemann_sampler import Riemann_sampler, riemann_plotter
from BQ_rays_subspace import BayesianQuadrature_rays, transform
from emukit.quadrature.methods import VanillaBayesianQuadrature, WarpedBayesianQuadratureModel, BoundedBayesianQuadrature
from emukit.quadrature.methods.warpings import SquareRootWarping
from emukit.model_wrappers import GPyModelWrapper
import numpy as np
import seaborn as sns
import pandas as pd
from classification_eval import eval_classification_loss

torch.manual_seed(0)

num_features=2
num_outputs=3
#model = LinearModel(num_features=num_features, num_outputs=1, bias = True)
model = FunctionApproximatorModel(num_features=num_features, hidden_layers=[50,50], num_outputs=num_outputs, nonlin = torch.nn.Tanh(), seed=47)

class SineModel_nd(torch.nn.Module):
    def __init__(self, num_features=1, num_outputs=1, bias=False):
        super(SineModel_nd, self).__init__()
        self.num_outputs = num_outputs
    def forward(self, x):
        out = torch.column_stack([(x + 1*torch.pi*2/self.num_outputs).sin().sum(dim=1)+1 for k in range(self.num_outputs)])
        return out

gen_model = SineModel_nd(num_features=num_features, num_outputs=num_outputs)
prior_sigma = 1.
target_sigma = 0.05

start_x = -2.5*2
end_x = 2.5*2
#input_dist = lambda x: torch.rand(x, num_features)*(end_x-start_x) + start_x
def input_dist(N):
    xs = torch.rand(N, num_features)*(end_x-start_x) + start_x
    return xs[:N]

def output_dist(y):
    return torch.distributions.Categorical(logits=y)

def output_func(y):
    logits = torch.nn.functional.softmax(y, dim=-1)
    return logits

loss_fn = NegLogLik_classification()

train_loader, test_loader = gen_model_data(gen_model, input_dist, num_train_samples=100, 
                                           num_test_samples=10, output_dist=output_dist, seed=2, batch_size=10)

pre_trained_params = torch.nn.utils.parameters_to_vector(model.parameters())

xs, ys = train_loader.dataset.dataset.tensors
xs_test, ys_test = test_loader.dataset.dataset.tensors

# Ok, we got some data now. 
# We already know the true mode, but because of the noise we have to post-train it, 
# to be sure that we are actually at the optimal parameters
# We do however start from the true parameters, so we should be able to get there

lr = torch.tensor(0.3)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
schedule_factor = 0.0001
epochs= 10000

scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=schedule_factor, total_iters=epochs)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=.995)

model, _, _, _, _ = train(model, train_loader=train_loader, test_loader=test_loader, optimizer=optimizer, 
              scheduler=scheduler, epochs=epochs, 
          prior_sigma=prior_sigma, loss_fn=loss_fn,
          device="cpu", logger_info=None,
          plot=False, plotpath=None, verbose = True, print_every_epoch=100, stop_lr=1e-6)

trained_params = torch.nn.utils.parameters_to_vector(model.parameters())

####################################
# Ok, we have trained the model. Evaluate point estimates
####################################
preds = output_func(model(xs_test))
print("\nEvaluating point estimate")
eval_res = eval_classification_loss(preds, ys_test)


####################################
# MCMC evaluation
####################################
parametersubset = dict(model.named_parameters())
sampler_mcmc = MCMC_sampler(model, parametersubset, xs=xs, ys=ys, loss_fn=neglog_loss(), prior_sigma=prior_sigma)
N_MCMC_samples = 10000
_=sampler_mcmc.make_posterior_sample(10000)

integral_mcmc, function_values_mcmc, weights_mcmc, posterior_samples_mcmc = integrator(sampler=sampler_mcmc, model_func=make_functional_fwd_xs(model), xs=xs_test, output_func=output_func)
means_mcmc = function_values_mcmc.mean(dim=0).detach()
upper95_mcmc = function_values_mcmc.quantile(.95, dim=0).detach()
lower95_mcmc = function_values_mcmc.quantile(.05, dim=0).detach()
print("\nEvaluating MCMC method")
eval_res = eval_classification_loss(means_mcmc, ys_test)



####################################
# Now we can try to fit a Laplace approximation to the posterior
####################################
subspace_rank=2
print(f"\nDoing Laplace approximation with subspace rank: {subspace_rank}")
laplace = Laplace(model, xs=xs, ys=ys, prior_sigma=prior_sigma, loss_fn=loss_fn, subspace_rank=subspace_rank)
_=laplace.fit_subspace(fitting_type="GGN", xs=xs, ys=ys)
N_Laplace_samples = 1000
laplace.make_posterior_sample(n_samples=N_Laplace_samples)
integral_la, function_values_la, weights_la, posterior_samples_la = integrator(sampler=laplace, model_func=make_functional_fwd_xs(model), xs=xs_test, output_func=output_func)
means_la = function_values_la.mean(dim=0).detach()
upper95_la = function_values_la.quantile(.95, dim=0).detach()
lower95_la = function_values_la.quantile(.05, dim=0).detach()
_=eval_classification_loss(means_la, ys_test)



####################################
# Riemann integration
####################################
R_sampler = Riemann_sampler(model, xs=xs, ys=ys, loss_fn=loss_fn, prior_sigma=prior_sigma,
                            n_posterior_samples=10, subspace_rank=subspace_rank)
_=R_sampler.fit(fitting_type="GGN")

_=R_sampler.make_posterior_sample(n_samples=1)

integral_riemann, function_values_riemann, weights, posterior_samples = integrator(R_sampler, model_func=make_functional_fwd_xs(model), xs=xs_test, output_func=output_func)
means_riemann = integral_riemann.detach()
epistemic_var = (function_values_riemann[:,:, :]-means_riemann).pow(2).mean(dim=0).detach()
upper95_riemann = function_values_riemann.quantile(.95, dim=0).detach()
lower95_riemann = function_values_riemann.quantile(.05, dim=0).detach()
_=eval_classification_loss(means_riemann, ys_test)


####################################
# Riemannian subspace integration
####################################
R_sampler = Riemann_sampler(model, xs=xs, ys=ys, loss_fn=loss_fn, prior_sigma=prior_sigma,
                            n_posterior_samples=10, subspace_rank=2)
_=R_sampler.fit(fitting_type="GGN")

for i in range(10):
    print(i)
    _=R_sampler.make_posterior_sample(n_samples=1)
#fig, ax = riemann_plotter(R_sampler, plot_traject=False, sample_markers=None)
#ax.scatter(R_sampler.posterior_samples_la[:, 0].detach(), R_sampler.posterior_samples_la[:, 1].detach(), color="k")
#plt.show()
integral_riemann, function_values_riemann, weights, posterior_samples = integrator(R_sampler, model_func=make_functional_fwd_xs(model), xs=xs_test, output_func=output_func)
means_riemann = integral_riemann.detach()
epistemic_var = (function_values_riemann[:,:, :]-means_riemann).pow(2).mean(dim=0).detach()
upper95_riemann = function_values_riemann.quantile(.95, dim=0).detach()
lower95_riemann = function_values_riemann.quantile(.05, dim=0).detach()
_=eval_classification_loss(means_riemann, ys_test)

####################################
# Bayesian Quadrature integration
####################################
R_sampler = Riemann_sampler(model, xs=xs, ys=ys, loss_fn=loss_fn, prior_sigma=prior_sigma, subspace_rank=2)
_=R_sampler.fit(fitting_type="GGN")

BQ = BayesianQuadrature_rays(R_sampler, evaluation_model=model, measure="gaussian_rescaled", integral_bounds_std=4, 
                             GP_lengthscale=1.0, GP_variance=1.0, num_timesteps=7, use_ray_acqusition=True, 
                             use_rays=True, 
                             theta_space_plot_limits=[[-1,1], [-1,1]], xs = xs[0], parametersubset=None, output_func=identity_func)

for i in range(16):
    integral_mean, integral_variance = BQ.step()
    print(f"{i = }, observations = {BQ.emukit_method.X.shape[0]}, {integral_mean = }, {integral_variance = }")

test_output_func = output_func

pred_samples = torch.tensor(BQ.pred_BQ_samples(xs_plt, 1000))

BQ_samp_trans = test_output_func(pred_samples)
BQ_samp_trans_mean = BQ_samp_trans.mean(dim=1)
BQ_samp_trans_95quant = BQ_samp_trans.quantile(.95, dim=1)
BQ_samp_trans_05quant = BQ_samp_trans.quantile(.05, dim=1)

for i in range(ys_plt_generated.unique().shape[0]):
    plt.plot(xs_plt, BQ_samp_trans_mean[:,i], c=c_list[i], label=f"posterior mean class {i}")
    plt.plot(xs_plt, ys_prob_plt_trained[:,i], c=c_list[i], label=f"true class {i}", linestyle="dashed")
    plt.fill_between(xs_plt[:,0], BQ_samp_trans_05quant[:,i], BQ_samp_trans_95quant[:,i], alpha=0.3, color=c_list[i], label=f"posterior std {i}")
plt.legend()
plt.show()

means_BQ, epistemic_var_BQ, vars = BQ.pred_BQ(xs_plt, get_measure_variance=True, transform_type="identity")
means_BQ_trans = test_output_func(means_BQ)
pred_dist = torch.distributions.Normal(means_BQ, epistemic_var_BQ.sqrt()).sample((10000,))
samp_trans = test_output_func(pred_dist)
mean_BQ = samp_trans.mean(dim=0)
lower_BQ = samp_trans.quantile(.05, dim=0)
upper_BQ = samp_trans.quantile(.95, dim=0)


#plt.scatter(xs_plt.unsqueeze(1).repeat([1, preds.shape[1], 1]), preds, marker="x", c="y", label="second central moment")
#plt.scatter(xs_plt, second_central_moments_mus - first_central_moments_mus**2, marker="x", c="g", label="first central moment")
for i in range(ys_plt_generated.unique().shape[0]):
    plt.plot(xs_plt, mean_BQ[:,i], c=c_list[i], label=f"posterior mean class {i}")
    plt.plot(xs_plt, ys_prob_plt_trained[:,i], c=c_list[i], label=f"true class {i}", linestyle="dashed")
    plt.fill_between(xs_plt[:,0], lower_BQ[:,i], upper_BQ[:,i], alpha=0.3, color=c_list[i], label=f"posterior std {i}")
plt.legend()
plt.show()

plt.plot(xs_plt, np.sqrt(epistemic_var_BQ))
plt.show()

#means_BQ_trans.sum(dim=1)
