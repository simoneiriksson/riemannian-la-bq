# this code takes a model and a dataset and computes the hessian of the loss function
#
import torch
from torch.func import grad, jvp, vjp, hessian, jacfwd, jacrev, vmap, functional_call



def hessian_from_func(func, x):
  """
  Compute the hessian of a function at a point x
  """
  H_func = hessian(func)
  H = H_func(x)
  return H

def make_functional_fwd(_model):
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

def hessian_from_model_loss_and_data(model, loss_func, xs, ys):
  model_functional = make_functional_fwd(model)  # get me the functional version of the model
  loss_functional = functional_loss(model_functional, loss_func)  # get me the functional version of the loss
  H_func = hessian(loss_functional, argnums=0)
  params = dict(model.named_parameters())
  H = H_func(params, xs, ys)
  return H


def hessian_dict_to_matrix(hess_dict, verbose=False, device="cpu"):
    hess_size=0
    # get parameter sizes:

    parameter_properties = {}

    for key1 in hess_dict.keys():
        mat = hess_dict[key1][key1]
        param_dims = len(mat.shape)//2
        param_shape = mat.shape[0:param_dims]
        param_numel = torch.prod(torch.tensor(param_shape)).item()
        param_name = key1
        parameter_properties[param_name] = {'param_shape': param_shape, 'param_dims': param_dims, 'param_numel': param_numel}
        hess_size += param_numel

    hess_matrix = torch.zeros((hess_size, hess_size), device=device)
    index1 = 0
    index2 = 0
    for key1 in hess_dict.keys():
        numel1 = parameter_properties[key1]['param_numel']
        dim1 = parameter_properties[key1]['param_dims']
        for key2 in hess_dict[key1].keys():
            if verbose:
                print(f"\n\n{key1 = }, {key2 = }")
                print(f"{index1 = }, {index2 = }")
            local_hess = hess_dict[key1][key2]
            if verbose:
                print(f"{local_hess.shape = }") 
                print(f"{local_hess = }")
                print(f"{parameter_properties[key1]['param_shape'] = }")
                print(f"{parameter_properties[key2]['param_shape'] = }")

            numel2 = parameter_properties[key2]['param_numel']
            dim2 = parameter_properties[key2]['param_dims']
            if verbose:
                print(f"{numel1 = }, {numel2 = }")
                print(f"{dim1 = }, {dim2 = }")

            flatten_key1 = torch.flatten(local_hess, start_dim=0, end_dim=dim1-1)
            if verbose:
                print(f"{flatten_key1.shape = }")
            local_hess_flat = torch.flatten(flatten_key1, start_dim=1)
            if verbose:
                print(f"{local_hess_flat.shape = }")

            hess_matrix[index1:index1+numel1, index2:index2+numel2] = local_hess_flat
            if verbose:
                print(f"{local_hess_flat = }")

            index2 += numel2
        index1 += numel1
        index2 = 0
    return hess_matrix, parameter_properties


if __name__ == "main":
  # run a test of hessian_from_func
  A = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
  B = torch.tensor([[1.0, 1.0]])
  C = torch.tensor([1.0])

  def second_degree(x):
    return x.T @ A @ x + B @ x + C

  x = torch.tensor([1.0, 1.0])
  print(f"{second_degree(x) = }")
  H = hessian_from_func(second_degree, x)
  print(f"{H = }")
  print(f"{A + A.T = }")  
  assert torch.allclose(H, A + A.T)

  # run a test of hessian_from_model_loss_and_data
  # we make a very simple model 
  class SimpleModel(torch.nn.Module):
    def __init__(self, feature_dim=2):
      super(SimpleModel, self).__init__()
      self.feature_dim = feature_dim
      self.lin1 = torch.nn.Linear(feature_dim, feature_dim, bias=False)
      self.lin2 = torch.nn.Linear(feature_dim, feature_dim, bias=False)
      torch.nn.init.constant_(self.lin1.weight, 2.0)
      torch.nn.init.constant_(self.lin2.weight, 3.0)

    def forward(self, x):
      return self.lin2(self.lin1(x)).sum(dim=1) 

  def sum_loss():
    def fn(preds, targets):
      return torch.sum(preds)
    return fn

  loss = sum_loss()
  xs = torch.tensor([[5.0]])
  ys = torch.tensor([1.0])

  model = SimpleModel(feature_dim=1)
  preds = model(xs)
  print(f"{preds = }")
  loss(preds, ys)
  H_dict = hessian_from_model_loss_and_data(model, loss, xs, ys)
  print(f"{H_dict = }")

  H, params = hessian_dict_to_matrix(H_dict)
  print(f"{H = }") # should return a 2x2 matrix, With 5. in the counterdiagonal and zero in the diagonal

  for param in params.keys():
    print(f"{param = }")
    print(f"{params[param] = }")

