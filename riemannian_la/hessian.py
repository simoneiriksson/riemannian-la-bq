"""
This module contains functions to compute the hessian of a function at a point, 
and the hessian of a model at a point
"""

import torch
from torch.func import grad, jvp, vjp, hessian, jacfwd, jacrev, vmap, functional_call


def hessian_from_func(func, x):
  """
  Compute the hessian of a function at a point x
  """
  H_func = hessian(func)
  H = H_func(x)
  return H

def make_functional_fwd_xs(_model):
  def fn(parameters, xs):
    return functional_call(_model, parameters, xs)
  return fn

def functional_loss(model_func, loss_func):
  # Returns a function that takes parameters, data, and target and returns the loss
  def fn(parameters, xs, ys):
    pred = model_func(parameters, xs)
    loss = loss_func(pred, ys)
    return loss
  return fn

def hessian_from_model_loss_and_data(model, parametersubset=None, loss_fn=None, xs=None, ys=None):
  model_functional = make_functional_fwd_xs(model)  # get me the functional version of the model
  if parametersubset is None:
    parametersubset = dict(model.named_parameters())
  else:
    parametersubset = parametersubset
  loss_functional = functional_loss(model_functional, loss_fn)  # get me the functional version of the loss
  H_func = hessian(loss_functional, argnums=0)
  H = H_func(parametersubset, xs, ys)
  return H

def hessian_from_loader(model, dataloader = None, loss_fn=None, parametersubset=None, device="cpu"):
    if parametersubset is None:
        parametersubset = dict(model.named_parameters())
    num_params =  sum([p.numel() for p in parametersubset.values()])
    ggn_hessian = torch.zeros((num_params, num_params), device=device)
    for i, (x, y) in enumerate(dataloader):  # loop over batches
        x_batch = x.to(device)
        y_batch = y.to(device)
        ggn_hessian += hessian_from_model_loss_and_data(model, x_batch, y_batch, parametersubset=parametersubset, loss_fn=loss_fn).detach().clone()
    return ggn_hessian

def hessian_dict_to_matrix(hess_dict, verbose=False, device="cpu"):
    hess_size=0
    parameter_properties = {}
    # The hessian dictionary is a dictionary of dictionaries. The outer dictionary has the parameter names as keys. 
    # The inner dictionary has the parameter names as keys and the hessian tensor object as values.
    # the hesian tensor object is a tensor that has the dimensionality = (parameter dimensionality + parameter dimensionality)
    # So if the parameter is the weight matrix of a linear layer, the hessian tensor object will have the shape (weight_dim, weight_dim, weight_dim, weight_dim)

    # Get parameter properties 
    for key1 in hess_dict.keys():
        mat = hess_dict[key1][key1]  # take the diagonal element. This element is a tensor that has the dimensionality = parameter dimensionality + parameter dimensionality 
        param_dims = len(mat.shape)//2  # get the number of dimensions of the parameter tensor
        param_shape = mat.shape[0:param_dims]  # get the shape of the parameter tensor (which is the first half of the dimensions of the diagonal element)
        param_numel = int(torch.prod(torch.tensor(param_shape)).item())  # get the number of elements in the parameter
        param_name = key1
        parameter_properties[param_name] = {'param_shape': param_shape, 'param_dims': param_dims, 'param_numel': param_numel}  # save information in dict
        hess_size += param_numel
    #print(hess_size)
    hess_matrix = torch.zeros((int(hess_size), int(hess_size)), device=device)
    index1 = 0
    index2 = 0
    # loop over the hessian dictionary and fill in the hessian matrix
    for key1 in hess_dict.keys():
        numel1 = parameter_properties[key1]['param_numel']
        dim1 = parameter_properties[key1]['param_dims']
        for key2 in hess_dict[key1].keys():
            local_hess = hess_dict[key1][key2]
            numel2 = parameter_properties[key2]['param_numel']
            dim2 = parameter_properties[key2]['param_dims']
            flatten_key1 = torch.flatten(local_hess, start_dim=0, end_dim=min(dim1-1,0))
            local_hess_flat = torch.flatten(flatten_key1, start_dim=min(dim2, 1))
            hess_matrix[index1:index1+numel1, index2:index2+numel2] = local_hess_flat
            index2 += numel2
        index1 += numel1
        index2 = 0
    return hess_matrix, parameter_properties

