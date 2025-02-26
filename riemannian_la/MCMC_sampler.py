import os
import sys
# set working directory
#os.chdir("../riemannian_la")
# print(os.getcwd())
from torch.utils.data import DataLoader, TensorDataset
from models import LinearModel, Model_from_func
from getdata import gen_log_regression_data
from train import train
from laplace_approx import Laplace, vector_to_parameterdict
from utils import make_functional_fwd, loss_func_from_target_sigma, make_functional_fwd_xs, functional_loss, functional_loss_for_vmap, sum_loss, neglog_loss, iid_gaussian_prior_loss
from discrete_sampler import discrete_function_sampler, discrete_model_sampler
from integration import integrator
from torch.func import grad, jvp, vjp, hessian, jacfwd, jacrev, vmap, functional_call
from matplotlib import pyplot as plt
from laplace_approx import Laplace
import torch
from models import functional_banana
import hamiltorch


# Now let us make a class that wraps around hamiltorch, that we can use for the project.
class MCMC_sampler():
    def __init__(self, model, parametersubset=None, dataloader=None, xs=None, ys=None,
                prior_sigma=None, prior_loss=None, target_sigma=None, loss_fn=None, device="cpu", verbose=False,
                n_posterior_samples=1000, step_size=0.3, num_steps_per_sample=5, sampler=hamiltorch.Sampler.HMC
                ):
        self.model = model
        self.functional_model = make_functional_fwd_xs(self.model)
        self.parametersubset = parametersubset
        self.device = device
        self.verbose = verbose
        self.is_fitted = False
        self.n_posterior_samples = n_posterior_samples
        self.dataloader = dataloader
        self.xs = xs
        self.ys = ys

        self.prior_sigma = prior_sigma
        self.prior_loss = prior_loss
        self.target_sigma = target_sigma
        self.loss_fn = loss_fn
        self.step_size = step_size
        self.num_steps_per_sample = num_steps_per_sample
        self.sampler = sampler
        if ((dataloader is not None) and (xs is not None)) or ((xs is None) and (dataloader is None)):
            raise ValueError("Either dataloader or xs must be provided")
        if xs is not None:
            self.dataloader = DataLoader(TensorDataset(xs, ys), batch_size=len(xs))
        else: self.dataloader = dataloader

        assert (prior_loss is None) ^ (prior_sigma is None), "Either prior_logprob or prior_sigma, but not both must be specified"
        assert (target_sigma is None) or (loss_fn is None), "Can't specify both target_sigma and loss_fn at the same time"

        self.loss_fn = loss_func_from_target_sigma(loss_fn, target_sigma)
        self.loss_func = functional_loss_for_vmap(self.functional_model, self.parametersubset, self.loss_fn, 
                                             self.xs, self.ys, prior_loss=self.prior_loss)
        self.neg_loss_func = lambda params: -self.loss_func(params)
        
        if self.prior_sigma is not None:
            if self.prior_sigma == 0:
                self.prior_loss = lambda pred, target: torch.tensor(0.0)
            else:
                self.prior_loss = lambda pred, target: iid_gaussian_prior_loss(prior_sigma=self.prior_sigma)(pred)

        if parametersubset is None:
            self.parametersubset = dict(model.named_parameters())
        else:
            self.parametersubset = parametersubset
        
        self.mean = torch.nn.utils.parameters_to_vector(self.parametersubset.values())
        self.num_params = self.mean.numel()

    def make_posterior_sample(self, n_samples=None):
        if n_samples is None:
            n_samples = self.n_posterior_samples


        params_hmc = hamiltorch.sample(log_prob_func=self.neg_loss_func, params_init=self.mean, num_samples=n_samples,
                                    step_size=self.step_size, num_steps_per_sample=self.num_steps_per_sample, sampler=self.sampler)
        self.posterior_samples = torch.stack(params_hmc)
        return self.posterior_samples









