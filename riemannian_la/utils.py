import torch
from torch.func import grad, jvp, vjp, hessian, jacfwd, jacrev, vmap, functional_call

def tensify(variable):
    if isinstance(variable, torch.Tensor):
        return variable
    elif variable==None:
        return None
    else: 
        return torch.tensor(variable, dtype=torch.float32)


def NegLogLik_regression(target_sigma=1.0):
    def fn(pred, target):
        #loss = (pred - target).pow(2).sum()/(2 * target_sigma**2)
        loss = torch.nn.MSELoss(reduction="sum")(pred, target)/(2 * tensify(target_sigma)**2)
        return loss
    return fn

def iid_gaussian_prior(prior_sigma=1.0):
    if prior_sigma == 0:  # if prior_sigma is zero, return a function that returns zero - that is: no regularization
        def fn(parameters):
            return torch.tensor(0.0)
        return
    def fn(parameters):
        return parameters.pow(2).sum()/(2 * prior_sigma**2)
    return fn

def NegLogLik_classification():
    def fn(pred, target):
        return torch.nn.CrossEntropyLoss(reduction="sum")(pred, target)
    return fn

def loss_func_from_target_sigma(loss_fn, target_sigma):
    if loss_fn is None and target_sigma is not None:  # assume regression
        loss_fn = NegLogLik_regression(target_sigma=target_sigma)

    if loss_fn is None and target_sigma is None:  # assume classification
        loss_fn = NegLogLik_classification()
    return loss_fn

def make_functional_fwd_xs(_model):
  def fn(parameters, xs):
    return functional_call(_model, parameters, xs)
  return fn

def make_functional_fwd(_model, xs):
    def fn(parameters):
        return functional_call(_model, parameters, (xs.unsqueeze(0),)).squeeze(0)
    return fn

def functional_loss(model_func, loss_func):
  # Returns a function that takes parameters, data, and target and returns the loss
  def fn(parameters, xs, ys):
    pred = model_func(parameters, xs)
    loss = loss_func(pred, ys)
    return loss
  return fn


def vector_to_parameterdict(vector, parametersubset=None):
    # Take a vector and return a dictionary of parameters with the same structure as parametersubset
    counter = 0
    parameter_dict = {}
    for key in parametersubset.keys():
        parameter_dict[key] = vector[counter:counter+parametersubset[key].numel()].view(parametersubset[key].shape)
        counter += parametersubset[key].numel()
    return parameter_dict



def sum_loss():
    def fn(preds, targets):
        return torch.sum(preds)
    return fn

def neglog_loss():
    def fn(preds, targets):
        return -torch.sum(preds.log())
    return fn

def functional_loss_for_vmap(model_func, parametersubset, loss_func, xs, ys):
    # Returns a function that takes parameters, data, and target and returns the loss
    def fn(parameters):
        param_dict = vector_to_parameterdict(parameters, parametersubset)
        pred = model_func(param_dict, xs)
        loss = loss_func(pred, ys)
        return loss
    return fn