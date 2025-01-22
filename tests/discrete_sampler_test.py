import os
import sys
# set working directory
os.chdir("../riemannian_la")
# print(os.getcwd())
from models import LinearModel, Model_from_func
from getdata import gen_log_regression_data
from train import train
from laplace_approx import Laplace, vector_to_parameterdict
from utils import loss_func_from_target_sigma, make_functional_fwd_xs, functional_loss, functional_loss_for_vmap, sum_loss, neglog_loss
from discrete_sampler import discrete_function_sampler, discrete_model_sampler
from integration import integrator
from torch.func import grad, jvp, vjp, hessian, jacfwd, jacrev, vmap, functional_call
from matplotlib import pyplot as plt
from laplace_approx import Laplace
import torch

#Let's do discrete integration with the banana function
from models import functional_banana
n_mesh = 100
#banana_function = functional_banana(curvature=2.0, sigma_x=2.0, sigma_y=1.0)
#neg_banana = lambda x: -functional_banana(curvature=0.0, sigma_x=1.0, sigma_y=1.0)(x)
banana_function = functional_banana(curvature=1.0, sigma_x=2.0, sigma_y=1.0)
x = torch.tensor([1.0, 1.0])
print(f"{banana_function(x) = }")
span = 10
discrete_sampler = discrete_function_sampler(func=banana_function, limits=[[-span, span], [-span, span]], n_mesh=n_mesh, normalize_weights=False)
posterior_samples, weights = discrete_sampler.samples_and_weights()

banana_function(posterior_samples[0])
sampler = discrete_sampler

xs = torch.tensor([0.0, 0.0]).unsqueeze(0)
ys = banana_function(xs[0]).unsqueeze(0)

posterior_samples, weights = discrete_sampler.samples_and_weights()

plt.scatter(posterior_samples[:,0], posterior_samples[:,1], c=weights)
plt.colorbar()
plt.show()

#################
# Now we do the same, but with model-version
#################

banana_function = functional_banana(curvature=1.0, sigma_x=2.0, sigma_y=1.0)
banana_model = Model_from_func(banana_function, input_shape=[2])
parametersubset = dict(banana_model.named_parameters())
x = torch.tensor([0.0, 0.0])
print(f"{banana_function(x) = }")
torch.nn.utils.vector_to_parameters(torch.tensor([0.0, 0.0]), banana_model.parameters())
print(f"{banana_model(x) = }")

xs = torch.tensor([0.0, 0.0]).unsqueeze(0)
ys = banana_function(xs[0]).unsqueeze(0)

span = 10
discrete_sampler = discrete_model_sampler(banana_model, loss_fn=sum_loss(), xs=xs, ys=ys, limits=[[-span, span], [-span, span]], n_mesh=n_mesh, normalize_weights=False, prior_sigma=0)
posterior_samples, weights = discrete_sampler.samples_and_weights()

plt.scatter(posterior_samples[:,0], posterior_samples[:,1], c=weights)
plt.colorbar()
plt.show()