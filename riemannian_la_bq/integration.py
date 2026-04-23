
import os
import sys
# set working directory
# os.chdir("../riemannian_la")
# print(os.getcwd())
from torch.func import grad, jvp, vjp, hessian, jacfwd, jacrev, vmap, functional_call
from matplotlib import pyplot as plt
import torch
from riemannian_la_bq.models import LinearModel, Model_from_func
from riemannian_la_bq.getdata import gen_log_regression_data
from riemannian_la_bq.train import train
from riemannian_la_bq.laplace_approx import Laplace, vector_to_parameterdict
from riemannian_la_bq.utils import loss_func_from_target_sigma, make_functional_fwd_xs, functional_loss, functional_loss_for_vmap, sum_loss, neglog_loss, identity_func
from riemannian_la_bq.discrete_sampler import discrete_function_sampler, discrete_model_sampler
from riemannian_la_bq.laplace_approx import Laplace

def integrator(sampler=None, model_func=None, parametersubset=None, xs=None, output_func=None):
    # if no output function defined, use identity function
    if output_func==None:
        output_func = identity_func

    if hasattr(sampler, "discrete_sampler"):
        posterior_samples, weights = sampler.samples_and_weights()
    else:
        posterior_samples = sampler.posterior_samples
        weights = torch.ones(posterior_samples.shape[0]).to(xs.device)/posterior_samples.shape[0]

    if parametersubset is None:
        parametersubset = dict(sampler.model.named_parameters())
    else:
        parametersubset = dict(parametersubset)
        
    # loop over posterior samples
    for sample_no, posterior_sample in enumerate(posterior_samples):
        param_dict = vector_to_parameterdict(posterior_sample, parametersubset)
        function_value = output_func(model_func(param_dict, xs))
        if sample_no == 0: 
            function_values = torch.zeros((len(posterior_samples), *function_value.shape)).to(xs.device)
        function_values[sample_no] = function_value 

    integral = (function_values * weights[:, None, None] ).sum(dim=0)
    return integral, function_values, weights, posterior_samples



