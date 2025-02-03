# Make two basic models:
# One that returns the pdf of a multivariate normal distribution
# One that returns the banana function
# a class that takes a function as an argument and returns a model

import numpy as np
import torch
from matplotlib import pyplot as plt
import torch.nn as nn
from utils import tensify

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
        return (normalization * torch.exp(exponent)).unsqueeze(0)
    return banana

def functional_d1(a, left_limit, right_limit):
    a = tensify(a)
    left_limit = tensify(left_limit)
    right_limit = tensify(right_limit)

    def indef_integral(x):
        val = a**(-5/6) * (1/3 * (a**(-1/6)*x).atan() - \
        1/6 * (3**(1/2) - 2*a**(-1/6)*x).atan() + \
        1/6 * (3**(1/2) + 2*a**(-1/6)*x).atan() - \
        1/4 * 3**(-1/2) * (- torch.log(a**(1/3) - 3**(1/2) * a**(1/6)*x + x**2) + torch.log(a**(1/3) + 3**(1/2)*a**(1/6)*x + x**2)))
        return val

    definite_integral = indef_integral(right_limit) - indef_integral(left_limit)

    def fn(x):
        return (1/(x**6 +a)) / definite_integral

    return fn


def functional_d1_2(a, left_limit, right_limit):
    a = tensify(a)
    left_limit = tensify(left_limit)
    right_limit = tensify(right_limit)

    def indef_integral(x):
        val = torch.atan(x/a**(1/2))/(a**(1/2))
        return val

    definite_integral = indef_integral(right_limit) - indef_integral(left_limit)

    def fn(x):
        return (1/(x**2 +a)) / definite_integral

    return fn


def functional_d1_lognorm(mu, sigma):
    mu = tensify(mu)
    sigma = tensify(sigma)
    def fn(x):
        return torch.exp(-1/2 * ((torch.log(x) - mu)/sigma)**2) / (x * sigma * torch.tensor(2 * torch.pi).sqrt())
    return fn

def functional_d1_halfcircle(a):
    a = tensify(a)

    def indef_integral(x):
        # 1/2 x sqrt(a^2 - x^2) + 1/2 a^2 tan^(-1)(x/sqrt(a^2 - x^2)) + constant
        val = 1/2 * x * (a**2 - x**2).sqrt() + 1/2 * a**2 * (x/(a**2 - x**2).sqrt()).atan()
        return val

    definite_integral = indef_integral(a) - indef_integral(-a)

    def fn(x):
        #return (a**2 - x**2).sqrt() / definite_integral * ((x**2 < a**2) * (x**2 > -a**2)).float()
        return torch.nan_to_num((a**2 - x**2).sqrt() / definite_integral, nan=0.0)

    return fn



def functional_d1_fourth_degree_poly():
    def indef_integral(x):
        val = 1/105 * (x**7 + 7*x**5 + 14*x**3 + 7*x)
        return val
    
    definite_integral = indef_integral(1) - indef_integral(-1)

    def fn(x):
        # define a foruth degree polynomial that is zero at -1, 1, and integrates to 1 over [-1, 1]
        y = (1 - x**2)**2 / definite_integral
        return y * (x**2 < 1) * (x**2 > -1)

    return fn

def functional_d1_normal(mu, sigma):
    mu = tensify(mu)
    sigma = tensify(sigma)
    def fn(x):
        return torch.exp(-1/2 * ((x - mu)/sigma)**2) / (sigma * torch.tensor(2 * torch.pi).sqrt())
    return fn


# a class that takes a function as an argument and returns a torch model
class Model_from_func(torch.nn.Module):
    def __init__(self, function, input_shape):
        super(Model_from_func, self).__init__()
        self.params = torch.nn.Parameter(torch.ones(input_shape))
        self.function = function

    def forward(self, x=None):
        return self.function(self.params)


class LinearModel(torch.nn.Module):
    def __init__(self, num_features=2, num_outputs=1, bias=False):
        super(LinearModel, self).__init__()
        self.lin = torch.nn.Linear(num_features, num_outputs, bias=bias)
        torch.nn.init.constant_(self.lin.weight, 0.0)
        torch.nn.init.constant_(self.lin.bias, 0.0)
    def forward(self, x):
        return self.lin(x)
