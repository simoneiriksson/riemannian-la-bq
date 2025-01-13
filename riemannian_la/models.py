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
