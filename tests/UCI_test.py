import os
import sys
# set working directory
os.chdir("../riemannian_la")
print(os.getcwd())

import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import matplotlib 
from models import LinearModel, FunctionApproximatorModel, SineModel
from train import train
from laplace_approx import Laplace
from MCMC_sampler import MCMC_sampler
from getdata import make_loaders, torch_seed, gen_model_data, gen_log_regression_data, get_dataloader_scipy
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
import torchmetrics
from classification_eval import eval_classification_loss

torch.manual_seed(0)
train_loader, test_loader, classes = get_dataloader_scipy("wine", train_share=.9 , batch_size=16)

num_features = next(iter(train_loader))[0].shape[1]
num_outputs=classes

#model = LinearModel(num_features=num_features, num_outputs=1, bias = True)
model = FunctionApproximatorModel(num_features=num_features, hidden_layers=[50,50], num_outputs=num_outputs, nonlin = torch.nn.Tanh(), seed=47)

prior_sigma = 1.

def output_dist(y):
    return torch.distributions.Categorical(logits=y)

def output_func(y):
    logits = torch.nn.functional.softmax(y, dim=-1)
    return logits

loss_fn = NegLogLik_classification()

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

xs, ys = train_loader.dataset.dataset.tensors
xs_test, ys_test = test_loader.dataset.dataset.tensors

evals = {}

####################################
# Ok, we have trained the model. Evaluate point estimates
####################################
preds = output_func(model(xs_test))
print("Evaluating point estimate")
eval_res = eval_classification_loss(preds, ys_test)
evals["Point estimate"] = {"samples": 1, "subspace_rank":None, "Results":eval_res}


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
print("Evaluating MCMC method")
eval_res = eval_classification_loss(means_mcmc, ys_test)
evals["MCMC"] = {"samples": N_MCMC_samples, "subspace_rank":None, "Results":eval_res}

####################################
# Now we can try to fit a Laplace approximation to the posterior
####################################
subspace_ranks = [2**i for i in range(8)] + [None]

for subspace_rank in subspace_ranks:
    print(f"Doing Laplace approximation with subspace rank: {subspace_rank}")
    laplace = Laplace(model, xs=xs, ys=ys, prior_sigma=prior_sigma, loss_fn=loss_fn, subspace_rank=subspace_rank)
    _=laplace.fit_subspace(fitting_type="GGN", xs=xs, ys=ys)
    N_Laplace_samples = 1000
    laplace.make_posterior_sample(n_samples=N_Laplace_samples)
    integral_la, function_values_la, weights_la, posterior_samples_la = integrator(sampler=laplace, model_func=make_functional_fwd_xs(model), xs=xs_test, output_func=output_func)
    means_la = function_values_la.mean(dim=0).detach()
    upper95_la = function_values_la.quantile(.95, dim=0).detach()
    lower95_la = function_values_la.quantile(.05, dim=0).detach()
    _=eval_classification_loss(means_la, ys_test)
    evals["Laplace"] = {"samples": N_MCMC_samples, "subspace_rank":subspace_rank, "Results":eval_res}

####################################
# Bayesian Quadrature integration
####################################
for subspace_rank in subspace_ranks:
    print(f"Doing BQ-Riemann Laplace approximation with subspace rank: {subspace_rank}")

    R_sampler = Riemann_sampler(model, xs=xs, ys=ys, loss_fn=loss_fn, prior_sigma=prior_sigma, subspace_rank=subspace_rank)
    _=R_sampler.fit(fitting_type="GGN")

    BQ = BayesianQuadrature_rays(R_sampler, evaluation_model=model, measure="gaussian_rescaled", integral_bounds_std=4, 
                                GP_lengthscale=1.0, GP_variance=1.0, num_timesteps=7, use_ray_acqusition=True, 
                                use_rays=True, 
                                theta_space_plot_limits=[[-1,1], [-1,1]], xs = xs[0], parametersubset=None, output_func=identity_func)

    for i in range(64):
        integral_mean, integral_variance = BQ.step()
        print(f"{i = }, samples = {BQ.emukit_method.X.shape[0]}, {integral_mean = }, {integral_variance = }")

        test_output_func = output_func
        pred_samples = torch.tensor(BQ.pred_BQ_samples(xs_test, 1000))

        BQ_samp_trans = test_output_func(pred_samples)
        BQ_samp_trans_mean = BQ_samp_trans.mean(dim=1)
        BQ_samp_trans_95quant = BQ_samp_trans.quantile(.95, dim=1)
        BQ_samp_trans_05quant = BQ_samp_trans.quantile(.05, dim=1)
        eval_res = eval_classification_loss(means_la, ys_test)
        N_BQ_Riemman_samples = BQ.emukit_method.X.shape[0]
        evals["BQ-Riemann"] = {"samples": N_BQ_Riemman_samples, "subspace_rank":subspace_rank, "Results":eval_res}




