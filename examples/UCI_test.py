import os
import sys
# set working directory

#root = ""
#os.chdir(f"{root}riemannian_la")
#print(os.getcwd())

import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import matplotlib 
from riemannian_la_bq.models import LinearModel, FunctionApproximatorModel, SineModel
from riemannian_la_bq.train import train
from riemannian_la_bq.laplace_approx import Laplace
from riemannian_la_bq.MCMC_sampler import MCMC_sampler
from riemannian_la_bq.getdata import make_loaders, torch_seed, gen_model_data, gen_log_regression_data, get_dataloader_scipy
from riemannian_la_bq.utils import NegLogLik_regression, NegLogLik_classification, identity_func
from riemannian_la_bq.integration import integrator
from riemannian_la_bq.utils import loss_func_from_target_sigma, make_functional_fwd_xs, functional_loss, functional_loss_for_vmap, sum_loss, neglog_loss, setup_logger
from riemannian_la_bq.discrete_sampler import discrete_function_sampler, discrete_model_sampler
from riemannian_la_bq.riemann_sampler import Riemann_sampler, riemann_plotter
from riemannian_la_bq.BQ_rays_subspace import BayesianQuadrature_rays, transform
from emukit.quadrature.methods import VanillaBayesianQuadrature, WarpedBayesianQuadratureModel, BoundedBayesianQuadrature
from emukit.quadrature.methods.warpings import SquareRootWarping
from emukit.model_wrappers import GPyModelWrapper
import numpy as np
import seaborn as sns
import pandas as pd
import torchmetrics
from riemannian_la_bq.classification_eval import eval_classification_loss
from datetime import datetime
import pickle
import pprint

base_directory = ".."
logger_info = setup_logger(base_directory, file_logging=True)
logger_info('Start logging')
logger_info(f"base_directory: {base_directory}")

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

torch.manual_seed(0)
train_loader, test_loader, classes = get_dataloader_scipy("wine", train_share=.5 , batch_size=16, select_features=None)

num_features = next(iter(train_loader))[0].shape[1]
num_outputs=classes

#model = LinearModel(num_features=num_features, num_outputs=1, bias = True)
model = FunctionApproximatorModel(num_features=num_features, hidden_layers=[num_features*2,num_features*2], num_outputs=num_outputs, nonlin = torch.nn.Tanh(), seed=47)
model.to(device)
logger_info(f"model size: {torch.nn.utils.parameters_to_vector(model.parameters()).shape[0]}")
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
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=.997)

model, _, _, _, _ = train(model, train_loader=train_loader, test_loader=test_loader, optimizer=optimizer, 
              scheduler=scheduler, epochs=epochs, 
          prior_sigma=prior_sigma, loss_fn=loss_fn,
          device=device, logger_info=logger_info,
          plot=False, plotpath=None, verbose = True, print_every_epoch=100, stop_lr=1e-10)

xs_test, ys_test = test_loader.dataset.dataset.tensors
xs_test, ys_test = xs_test.to(device), ys_test.to(device)

xs_train, ys_train = train_loader.dataset.dataset.tensors
xs_train, ys_train = xs_train.to(device), ys_train.to(device)

k=6
subspace_ranks = [2**i for i in range(k)] + [None]
max_BQ_riemann_samples = 100
evals = []

# ####################################
# # Ok, we have trained the model. Evaluate point estimates
# ####################################
preds = output_func(model(xs_test))
logger_info(f"\n\nEvaluating point estimate")
eval_res = eval_classification_loss(preds, ys_test, logger=logger_info, device=device)
evals.append({"name": "Point estimate", "samples": 1, "subspace_rank":None, "Results":eval_res})


# ####################################
# # MCMC evaluation
# ####################################
parametersubset = dict(model.named_parameters())
sampler_mcmc = MCMC_sampler(model, parametersubset, xs=xs_train, ys=ys_train, loss_fn=neglog_loss(), prior_sigma=prior_sigma, device=device)
N_MCMC_samples = 1000
_=sampler_mcmc.make_posterior_sample(N_MCMC_samples)

integral_mcmc, function_values_mcmc, weights_mcmc, posterior_samples_mcmc = integrator(sampler=sampler_mcmc, model_func=make_functional_fwd_xs(model), xs=xs_test, output_func=output_func)
means_mcmc = function_values_mcmc.mean(dim=0).detach()
upper95_mcmc = function_values_mcmc.quantile(.95, dim=0).detach()
lower95_mcmc = function_values_mcmc.quantile(.05, dim=0).detach()
logger_info(f"\n\nEvaluating MCMC method")
eval_res = eval_classification_loss(means_mcmc, ys_test, logger=logger_info, device=device)
evals.append({"name": "MCMC", "samples": N_MCMC_samples, "subspace_rank":None, "Results":eval_res})

# ####################################
# # Now we can try to fit a Laplace approximation to the posterior
# ####################################
for subspace_rank in subspace_ranks:
    logger_info(f"\n\nDoing Laplace approximation with subspace rank: {subspace_rank}")
    laplace = Laplace(model, xs=xs_train, ys=ys_train, prior_sigma=prior_sigma, loss_fn=loss_fn, subspace_rank=subspace_rank, device=device)
    _=laplace.fit_subspace(fitting_type="GGN", xs=xs_train, ys=ys_train)
    N_Laplace_samples = 10000
    laplace.make_posterior_sample(n_samples=N_Laplace_samples)
    integral_la, function_values_la, weights_la, posterior_samples_la = integrator(sampler=laplace, model_func=make_functional_fwd_xs(model), xs=xs_test, output_func=output_func)
    means_la = function_values_la.mean(dim=0).detach()
    upper95_la = function_values_la.quantile(.95, dim=0).detach()
    lower95_la = function_values_la.quantile(.05, dim=0).detach()
    eval_res=eval_classification_loss(means_la, ys_test, logger=logger_info, device=device)
    evals.append({"name": "Laplace", "samples": N_MCMC_samples, "subspace_rank":subspace_rank, "Results":eval_res})


# ####################################
# # Riemannian subspace integration
# ####################################
for subspace_rank in subspace_ranks:
    R_sampler = Riemann_sampler(model, xs=xs_train, ys=ys_train, loss_fn=loss_fn, prior_sigma=prior_sigma,
                                n_posterior_samples=10, subspace_rank=subspace_rank, device=device)
    _=R_sampler.fit(fitting_type="GGN")

    logger_info(f"\n\nDoing Riemannian Laplace approximation with subspace rank: {subspace_rank}")
    for i in range(max_BQ_riemann_samples):
        logger_info(f"\nSampling {i}")
        _=R_sampler.make_posterior_sample(n_samples=10)
        integral_riemann, function_values_riemann, weights, posterior_samples = integrator(R_sampler, model_func=make_functional_fwd_xs(model), xs=xs_test, output_func=output_func)
        means_riemann = integral_riemann.detach()
        epistemic_var = (function_values_riemann[:,:, :]-means_riemann).pow(2).mean(dim=0).detach()
        upper95_riemann = function_values_riemann.quantile(.95, dim=0).detach()
        lower95_riemann = function_values_riemann.quantile(.05, dim=0).detach()
        eval_res=eval_classification_loss(means_riemann, ys_test, logger=logger_info, device=device)
        evals.append({"name": "Riemann", "samples": R_sampler.posterior_samples.shape[0], "subspace_rank":subspace_rank, "Results":eval_res})

####################################
# Bayesian Quadrature integration
####################################
for subspace_rank in subspace_ranks:
    if subspace_rank == None: continue
    logger_info(f"\n\nDoing BQ-Riemann Laplace approximation with subspace rank: {subspace_rank}")

    R_sampler = Riemann_sampler(model, xs=xs_train, ys=ys_train, loss_fn=loss_fn, prior_sigma=prior_sigma, subspace_rank=subspace_rank, device=device)
    _=R_sampler.fit(fitting_type="GGN")

    BQ = BayesianQuadrature_rays(R_sampler, evaluation_model=model, measure="gaussian_rescaled", integral_bounds_std=4, 
                                GP_lengthscale=1.0, GP_variance=1.0, num_timesteps=7, use_ray_acqusition=True, 
                                use_rays=True, 
                                theta_space_plot_limits=[[-1,1], [-1,1]], xs = xs_train[0], parametersubset=None, output_func=identity_func, device=device)
    #max_BQ_riemann_samples_ = min(max_BQ_riemann_samples_, 3**subspace_rank-1)
    for i in range(max_BQ_riemann_samples):
        for k in range(10):
            integral_mean, integral_variance = BQ.step()
        logger_info(f"\n{i = }, samples = {BQ.emukit_method.X.shape[0]}, {integral_mean = }, {integral_variance = }")

        test_output_func = output_func
        pred_samples = BQ.pred_BQ_samples(xs_test, 10000)

        BQ_samp_trans = test_output_func(pred_samples)
        BQ_samp_trans_mean = BQ_samp_trans.mean(dim=1)
        BQ_samp_trans_95quant = BQ_samp_trans.quantile(.95, dim=1)
        BQ_samp_trans_05quant = BQ_samp_trans.quantile(.05, dim=1)
        eval_res = eval_classification_loss(BQ_samp_trans_mean, ys_test, logger=logger_info, device=device)
        N_BQ_Riemman_samples = BQ.emukit_method.X.shape[0]
        #evals["BQ-Riemann"] = {"samples": N_BQ_Riemman_samples, "subspace_rank":subspace_rank, "Results":eval_res}
        evals.append({"name": "BQ-Riemann", "samples": N_BQ_Riemman_samples, "subspace_rank":subspace_rank, "Results":eval_res})

logger_info("Done with evaluations, now saving")


stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
filename = f"{base_directory}/evaluations/experiment_eval_{stamp}.pkl"
with open(filename, 'wb') as handle:
    pickle.dump(evals, handle)

pp = pprint.PrettyPrinter(depth=100)

with open(f"{base_directory}/evaluations/experiment_eval_{stamp}.txt", "w") as filename2:
    #print(pp.pprint(results), file=filename2)
    filename2.write(pp.pformat(evals))
    filename2.close()
