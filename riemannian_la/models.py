# Make two basic models:
# One that returns the pdf of a multivariate normal distribution
# One that returns the banana function
# a class that takes a function as an argument and returns a model

import numpy as np
import torch
from hessian import hessian_from_func, make_functional_fwd, functional_loss, hessian_from_model_loss_and_data, hessian_dict_to_matrix
from matplotlib import pyplot as plt
import torch.nn as nn

def functional_banana(curvature=2.0, sigma_x=2.0, sigma_y=1.0):
    def banana(input_2d):
        y = input_2d[1]
        x = input_2d[0]
        # Compute the transformed y
        y_transformed = y - curvature * x**2
        # Evaluate the 2D Gaussian PDF
        normalization = torch.tensor(1 / (2 * torch.pi * sigma_x * sigma_y))
        exponent = -0.5 * ((x / sigma_x)**2 + (y_transformed / sigma_y)**2)
        #return torch.log(normalization) + exponent
        return normalization * torch.exp(exponent)
    return banana

# a class that takes a function as an argument and returns a torch model
class Model_from_func(torch.nn.Module):
    def __init__(self, function, input_shape):
        super(Model_from_func, self).__init__()
        self.params = torch.nn.Parameter(torch.ones(input_shape))
        self.function = function

    def forward(self, x=None):
        return self.function(self.params)


class LinearModel(torch.nn.Module):
    def __init__(self, num_features=1, num_output=1, bias=False):
        super(LinearModel, self).__init__()
        self.lin = torch.nn.Linear(num_features, 1, bias=True)
        torch.nn.init.constant_(self.lin.weight, 1.0)
        torch.nn.init.constant_(self.lin.bias, 2.0)
    def forward(self, x):
        return self.lin(x)


class LogregModel(torch.nn.Module):
    def __init__(self, num_features=2, num_classes=2, bias=True):
        super(LogregModel, self).__init__()
        self.lin = torch.nn.Linear(num_features, 1, bias=bias)
        torch.nn.init.constant_(self.lin.weight, 1.0)
        torch.nn.init.constant_(self.lin.bias, 2.0)
    def forward(self, x):
        return self.lin(x)

if __name__ == "__main__":
    # testing the models

    def sum_loss():
        def fn(preds, targets):
            return torch.sum(preds)
        return fn
    def plot_2d_func(func, x_range, y_range, num_points=100):
        xs = np.linspace(x_range[0], x_range[1], num_points)
        ys = np.linspace(y_range[0], y_range[1], num_points)
        X, Y = np.meshgrid(xs, ys)
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = func(torch.tensor([X[i, j], Y[i, j]]))
        plt.contour(X, Y, Z, levels=100)
        plt.show()

    # define a "banana" function
    banana_function = functional_banana(curvature=2.0, sigma_x=2.0, sigma_y=1.0)

    # plot it
    plot_2d_func(banana_function, x_range=[-1, 1], y_range=[-1, 1])

    x = torch.tensor([0.0, 0.0])
    y = banana_function(x)

    # Take hessian in the point x
    print(f"{hessian_from_func(banana_function, torch.tensor([0.0, 0.0])) =}")
    # should return tensor([[-0.0199,  0.0000],[ 0.0000, -0.0796]])

    # Now make a model from the function. This model return the banana function, 
    # as function of the parameters in the model. 
    # That is, the model is a function of the parameters and not the input
    # Here we use a "loss" function that sums the output of the model
    banana_model = Model_from_func(banana_function, input_shape=[2])
    torch.nn.utils.vector_to_parameters(x, [banana_model.params])
    hess_dict = hessian_from_model_loss_and_data(banana_model, sum_loss(), x, y)
    # print(f"{hess_dict = }")

    # Convert the hessian dictionary to a matrix
    hess_matrix, params = hessian_dict_to_matrix(hess_dict)
    print(f"{hess_matrix = }") # should return  tensor([[-0.0199,  0.0000],[ 0.0000, -0.0796]])

    # Now, do the same, with a model that returns the square root of the banana function
    # And then use the MSELoss between the target and 0 (which just takes the square of the target)
    sqrt_root_banana = lambda x: functional_banana(curvature=2.0, sigma_x=2.0, sigma_y=1.0)(x)**.5
    sqrt_root_banana(x)
    mseloss = torch.nn.MSELoss()

    #mseloss(sqrt_root_banana(x), torch.tensor(0.0))
    sqrt_banana_model = Model_from_func(sqrt_root_banana, input_shape=[2])
    torch.nn.utils.vector_to_parameters(x, [sqrt_banana_model.params])
    hess_dict = hessian_from_model_loss_and_data(sqrt_banana_model, mseloss, x, torch.tensor(0.0))
    hess_matrix, params = hessian_dict_to_matrix(hess_dict)
    print(f"{hess_matrix = }") # should return  tensor([[-0.0199,  0.0000],[ 0.0000, -0.0796]])

    # Plot it to see if it looks like a banana
    plot_2d_func(sqrt_root_banana, x_range=[-1, 1], y_range=[-1, 1])

