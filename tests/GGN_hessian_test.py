import os
import sys
# set working directory
os.chdir("../riemannian_la")
from models import LinearModel
from matplotlib import pyplot as plt
from getdata import gen_linear_regression_data
import torch
from GGN_hessian import GGN_hessian
from hessian import hessian_from_model_loss_and_data, hessian_dict_to_matrix, hessian_from_func, hessian_from_model_loss_and_data, hessian_dict_to_matrix
from utils import tensify
from models import Model_from_func
from models import functional_banana
from utils import NegLogLik_regression

# Tests of function version of GGN_hessian on a simple linear regression model:
# define a simple model
model = LinearModel(num_features=1, num_outputs=1, bias=True)
# get some data
target_sigma = 0.1
train_loader, test_loader = gen_linear_regression_data(num_train_samples=10, num_test_samples=10, target_sigma=target_sigma, batch_size=0, seed=5)
xs, ys = next(iter(train_loader))

# Analytical solution:
Phi = torch.cat([xs, torch.ones(xs.shape[0],1)], dim=1)
prior_Sigma = torch.eye(2)
prior_mu = torch.zeros(2)

S_inverse = prior_Sigma.inverse() + target_sigma**-2 * Phi.T @ Phi
mu = S_inverse.inverse() @ (prior_Sigma.inverse() @ prior_mu + target_sigma**-2 * (Phi.T @ ys.squeeze()))
print(f"Analytical solution:")
print(f"{mu=}")
print(f"{S_inverse=}")

# GGN matrix solution:
torch.nn.utils.vector_to_parameters(mu, model.parameters())
GGN_matrix = GGN_hessian(model, xs, ys, loss_fn="regression", target_sigma=target_sigma)
GGN_matrix_post = GGN_matrix + prior_Sigma.inverse()
print(f"GGN solution with build-in loss-hessian:")
print(f"{GGN_matrix_post=}")


loss_fn = NegLogLik_regression(target_sigma=target_sigma)
GGN_matrix2 = GGN_hessian(model, xs, ys, loss_fn=loss_fn, target_sigma=target_sigma)
GGN_matrix_post2 = GGN_matrix2 + prior_Sigma.inverse()

print(f"GGN solution with loss-hessian derived from loss function:")
print(f"{GGN_matrix_post2=}")

# direct hessian computation:
hessian_dict = hessian_from_model_loss_and_data(model, loss_fn=loss_fn, xs=xs, ys=ys)
hessian_matrix, params = hessian_dict_to_matrix(hessian_dict)
hessian_matrix_post = hessian_matrix + prior_Sigma.inverse()

print(f"Solution with direct Hessian:")
print(f"{hessian_matrix_post=}")

#######################################################################################
#
# Now we try with a model that has a non-linear output, given by a function.
#
#######################################################################################

# run a test of hessian_from_model_loss_and_data
# define a "banana" function
banana_function = functional_banana(curvature=2.0, sigma_x=2.0, sigma_y=1.0)

x = torch.tensor([0.0, 0.0])
y = banana_function(x)
# Take hessian in the point x
print(f"{hessian_from_func(banana_function, torch.tensor([0.0, 0.0])) =}")
# should return tensor([[-0.0199,  0.0000],[ 0.0000, -0.0796]])

def sum_loss():
    def fn(preds, targets):
        return torch.sum(preds, dim=0)
    return fn

banana_model = Model_from_func(banana_function, input_shape=[2])
torch.nn.utils.vector_to_parameters(x, [banana_model.params])
hess_dict = hessian_from_model_loss_and_data(banana_model, sum_loss(), x, y)
# Convert the hessian dictionary to a matrix
hess_matrix, params = hessian_dict_to_matrix(hess_dict)
print(f"{hess_matrix = }") # should return  tensor([[-0.0199,  0.0000],[ 0.0000, -0.0796]])

GGN_matrix2 = GGN_hessian(banana_model, x, y, loss_fn=sum_loss(), target_sigma=target_sigma)
print(f"{GGN_matrix2=}")
# This should return 0-matrix, since the loss is linear, and the GGN uses the hessian of the loss


# Now, do the same, with a model that returns the square root of the banana function
# And then use the MSELoss between the target and 0 (which just takes the square of the target)
sqrt_root_banana = lambda x: functional_banana(curvature=2.0, sigma_x=2.0, sigma_y=1.0)(x)**.5
sqrt_root_banana(x)
mseloss = torch.nn.MSELoss(reduction="sum")

sqrt_banana_model = Model_from_func(sqrt_root_banana, input_shape=[2])
torch.nn.utils.vector_to_parameters(x, [sqrt_banana_model.params])
hess_dict = hessian_from_model_loss_and_data(sqrt_banana_model, mseloss, x, torch.tensor(0.0))
hess_matrix, params = hessian_dict_to_matrix(hess_dict)
print(f"{hess_matrix = }") # should return  tensor([[-0.0199,  0.0000],[ 0.0000, -0.0796]])
# Hurraaay! It works!

