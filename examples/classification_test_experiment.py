
import os
import sys
# set working directory

# root = ".."
# os.chdir(f"{root}/riemannian_la")
# print(os.getcwd())

import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import matplotlib 
from models import LinearModel, FunctionApproximatorModel, SineModel
from train import train
from laplace_approx import Laplace
from MCMC_sampler import MCMC_sampler
from getdata import make_loaders, torch_seed, gen_model_data, gen_log_regression_data, get_dataloader_scipy
from riemannian_la_bq.utils import NegLogLik_regression, NegLogLik_classification, identity_func
from integration import integrator
from riemannian_la_bq.utils import loss_func_from_target_sigma, make_functional_fwd_xs, functional_loss, functional_loss_for_vmap, sum_loss, neglog_loss, setup_logger
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
from datetime import datetime
import pickle
import pprint

base_directory = "."
logger_info = setup_logger(base_directory, file_logging=True)
logger_info('Start logging')
logger_info('Running synthetic regression experiment')
logger_info(f"base_directory: {base_directory}")

experiment_name = "classification_experiment4"

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"


torch.manual_seed(0)

num_features=1
num_outputs=3
#model = LinearModel(num_features=num_features, num_outputs=1, bias = True)
model = FunctionApproximatorModel(num_features=num_features, hidden_layers=[50,50], num_outputs=num_outputs, nonlin = torch.nn.Tanh(), seed=47)
model.to(device)


class SineModel(torch.nn.Module):
    def __init__(self, num_features=1, num_outputs=1, bias=False):
        super(SineModel, self).__init__()
        self.num_outputs = num_outputs
    def forward(self, x):
        out = torch.column_stack([(x + k*torch.pi*2/self.num_outputs).sin() for k in range(self.num_outputs)])*1.5
        return out

gen_model = SineModel(num_outputs=num_outputs)
gen_model.to(device)

prior_sigma = 1.
target_sigma = 0.05

start_x = -2.5*2
end_x = 2.5*2
#input_dist = lambda x: torch.rand(x, num_features)*(end_x-start_x) + start_x
def input_dist(N):
    xs = torch.rand(N*10, num_features)*(end_x-start_x) + start_x
    index = (1-((xs > -.0) & (xs < .0)).int()).nonzero()[:,0]
    return xs[index][:N]

def output_dist(y):
    return torch.distributions.Categorical(logits=y)

def output_func(y):
    logits = torch.nn.functional.softmax(y, dim=-1)
    return logits

#xs = input_dist(100).to(device)
#gen_model(xs)
loss_fn = NegLogLik_classification()

train_loader, test_loader = gen_model_data(gen_model, input_dist, num_train_samples=100, 
                                           num_test_samples=100, output_dist=output_dist, seed=2, batch_size=10)

pre_trained_params = torch.nn.utils.parameters_to_vector(model.parameters())
#xs, ys = train_loader.dataset.dataset.tensors
xs_plt = torch.linspace(start_x, end_x, 100).unsqueeze(1).to(device)

xs_test, ys_test = test_loader.dataset.dataset.tensors
xs_test, ys_test = xs_test.to(device), ys_test.to(device)

xs_train, ys_train = train_loader.dataset.dataset.tensors
xs_train, ys_train = xs_train.to(device), ys_train.to(device)

plt.scatter(xs_train[:, 0].detach().cpu(), ys_train.detach().cpu(), c="b", label="data")
#plt.show()

ys_prob_plt_generated = output_func(gen_model(xs_plt)).detach()
ys_plt_generated = output_dist(gen_model(xs_plt)).sample().detach()
c_list = ["red","green","blue"]
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", c_list)
plt.scatter(xs_train[:, 0].detach().cpu(), ys_train.cpu(), alpha=.1, c=ys_train.cpu(), cmap=cmap)
for i in range(ys_plt_generated.unique().shape[0]):
    plt.plot(xs_plt[:, 0].detach().cpu(), ys_prob_plt_generated[:,i].cpu(), color=c_list[i])
#plt.show()
plt.close()

k=6
subspace_ranks = [2**i for i in range(k)] + [None]
#subspace_ranks = [1, 2, 3, 4, 5, 6, 7, 8, None]

max_BQ_riemann_samples = 50
evals = []


# Ok, we got some data now. 
# We already know the true mode, but because of the noise we have to post-train it, 
# to be sure that we are actually at the optimal parameters
# We do however start from the true parameters, so we should be able to get there

lr = torch.tensor(0.1)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
schedule_factor = 0.0001
epochs= 10000

scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=schedule_factor, total_iters=epochs)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=.998)

model, _, _, _, _ = train(model, train_loader=train_loader, test_loader=test_loader, optimizer=optimizer, 
              scheduler=scheduler, epochs=epochs, 
          prior_sigma=prior_sigma, loss_fn=loss_fn,
          device=device, logger_info=logger_info,
          plot=False, plotpath=None, verbose = True, print_every_epoch=100, stop_lr=1e-5)

trained_params = torch.nn.utils.parameters_to_vector(model.parameters())
ys_prob_plt_trained = model(xs_plt).softmax(dim=1).detach()
ys_plt_trained = output_dist(model(xs_plt)).sample().detach()
plt.scatter(xs_train[:, 0].cpu(), ys_train.cpu(), alpha=.1, c=ys_train.cpu(), cmap=cmap)
#plt.scatter(xs[:, 0].detach(), ys, alpha=.1, c=ys, cmap=cmap)
for i in range(ys_plt_trained.unique().shape[0]):
    plt.plot(xs_plt[:, 0].cpu().detach(), ys_prob_plt_trained[:,i].cpu(), color=c_list[i])
#plt.show()
plt.close()




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

integral_mcmc, function_values_mcmc, weights_mcmc, posterior_samples_mcmc = integrator(sampler=sampler_mcmc, model_func=make_functional_fwd_xs(model), xs=xs_plt, output_func=output_func)
means_mcmc = function_values_mcmc.mean(dim=0).detach()
upper95_mcmc = function_values_mcmc.quantile(.95, dim=0).detach()
lower95_mcmc = function_values_mcmc.quantile(.05, dim=0).detach()
for i in range(ys_plt_generated.unique().shape[0]):
    plt.plot(xs_plt.cpu(), means_mcmc[:,i].cpu(), c=c_list[i], label=f"mean class {i}")
    plt.fill_between(xs_plt[:,0].cpu(), lower95_mcmc[:,i].cpu(),  upper95_mcmc[:,i].cpu(), alpha=0.3, color=c_list[i], label=f"epistemic class {i}")
    plt.plot(xs_plt[:, 0].cpu(), ys_prob_plt_trained[:,i].cpu(), color=c_list[i], linestyle="dashed", label="true prob")    
#plt.legend()
plt.savefig(f"{base_directory}/figures/{experiment_name}/MCMC_samples{N_MCMC_samples}.png")
#plt.show()
plt.close()


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
    evals.append({"name": "Laplace", "samples": N_Laplace_samples, "subspace_rank":subspace_rank, "Results":eval_res})


    integral_la, function_values_la, weights_la, posterior_samples_la = integrator(sampler=laplace, model_func=make_functional_fwd_xs(model), xs=xs_plt, output_func=output_func)
    means_la = function_values_la.mean(dim=0).detach()
    upper95_la = function_values_la.quantile(.95, dim=0).detach()
    lower95_la = function_values_la.quantile(.05, dim=0).detach()

    for i in range(ys_plt_generated.unique().shape[0]):
        plt.plot(xs_plt.cpu(), means_la[:,i].cpu(), c=c_list[i], label=f"mean class {i}")
        plt.fill_between(xs_plt[:,0].cpu(), lower95_la[:,i].cpu(),  upper95_la[:,i].cpu(), alpha=0.3, color=c_list[i], label=f"epistemic class {i}")
        plt.plot(xs_plt[:, 0].cpu(), ys_prob_plt_trained[:,i].cpu(), color=c_list[i], linestyle="dashed", label="true prob")    
    #plt.legend()
    plt.savefig(f"{base_directory}/figures/{experiment_name}/laplace_subspace_rank_{subspace_rank}_samples{N_Laplace_samples}.png")
    plt.show()
    plt.close()

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

    integral_riemann, function_values_riemann, weights, posterior_samples = integrator(R_sampler, model_func=make_functional_fwd_xs(model), xs=xs_plt, output_func=output_func)
    means_riemann = function_values_riemann.mean(dim=0)
    upper95_riemann = function_values_riemann.quantile(.95, dim=0)
    lower95_riemann = function_values_riemann.quantile(.05, dim=0)
    for i in range(ys_plt_generated.unique().shape[0]):
        plt.plot(xs_plt.cpu(), means_riemann[:,i].cpu(), c=c_list[i], label=f"mean class {i}")
        plt.fill_between(xs_plt[:,0].cpu(), lower95_riemann[:,i].cpu(),  upper95_riemann[:,i].cpu(), alpha=0.3, color=c_list[i], label=f"epistemic class {i}")
        plt.plot(xs_plt[:, 0].cpu(), ys_prob_plt_trained[:,i].cpu(), color=c_list[i], linestyle="dashed", label="true prob")    
    #plt.legend()
    plt.savefig(f"{base_directory}/figures/{experiment_name}/riemannian_subspace_rank_{subspace_rank}_samples{max_BQ_riemann_samples}.png")
    #plt.show()
    plt.close()




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
    if subspace_rank == 1: 
        BQ_riemann_samples = 1
    else: 
        BQ_riemann_samples = max_BQ_riemann_samples
    for i in range(BQ_riemann_samples):
        logger_info(f"\n{i = }")
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

    pred_samples = torch.tensor(BQ.pred_BQ_samples(xs_plt, 1000))
    BQ_samp_trans = test_output_func(pred_samples)
    BQ_samp_trans_mean = BQ_samp_trans.mean(dim=1)
    BQ_samp_trans_95quant = BQ_samp_trans.quantile(.95, dim=1)
    BQ_samp_trans_05quant = BQ_samp_trans.quantile(.05, dim=1)
    for i in range(ys_plt_generated.unique().shape[0]):
        plt.plot(xs_plt.cpu(), BQ_samp_trans_mean[:,i].cpu(), c=c_list[i], label=f"posterior mean class {i}")
        plt.plot(xs_plt.cpu(), ys_prob_plt_trained[:,i].cpu(), c=c_list[i], label=f"true class {i}", linestyle="dashed")
        plt.fill_between(xs_plt[:,0].cpu(), BQ_samp_trans_05quant[:,i].cpu(), BQ_samp_trans_95quant[:,i].cpu(), alpha=0.3, color=c_list[i], label=f"posterior std {i}")
    #plt.legend()
    plt.savefig(f"{base_directory}/figures/{experiment_name}/riemannian_BQ_subspace_rank_{subspace_rank}_samples{BQ_riemann_samples}.png")
    #plt.show()
    plt.close()
    


logger_info("Done with evaluations, now saving")


stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
filename = f"{base_directory}/evaluations/synth_experiment_eval_{stamp}.pkl"
with open(filename, 'wb') as handle:
    pickle.dump(evals, handle)

pp = pprint.PrettyPrinter(depth=100)

with open(f"{base_directory}/evaluations/synth_experiment_eval_{stamp}.txt", "w") as filename2:
    #print(pp.pprint(results), file=filename2)
    filename2.write(pp.pformat(evals))
    filename2.close()
