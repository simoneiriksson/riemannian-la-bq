import torch
from torch.func import grad, jvp, vjp, hessian, jacfwd, jacrev, vmap, functional_call
from contextlib import contextmanager
import logging
from datetime import datetime
import os

def tensify(variable):
    if isinstance(variable, torch.Tensor):
        return variable
    elif type(variable) == type(None):
        return None
    else: 
        return torch.tensor(variable, dtype=torch.float32)


def iid_gaussian_prior_loss(prior_sigma=1.0):
    if prior_sigma == 0:  # if prior_sigma is zero, return a function that returns zero - that is: no regularization
        def fn(parameters):
            return torch.tensor(0.0)
        return
    def fn(parameters):
        return parameters.pow(2).sum()/(2 * prior_sigma**2)
    return fn

def NegLogLik_regression(target_sigma=1.0):
    def fn(pred, target):
        #loss = (pred - target).pow(2).sum()/(2 * target_sigma**2)
        loss = torch.nn.MSELoss(reduction="sum")(pred, target)/(2 * tensify(target_sigma)**2)
        return loss
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


def identity_func(x):
    return x

def make_functional_fwd_xs(_model, output_func=None):
  if output_func == None: output_func=identity_func
  def fn(parameters, xs):
    return output_func(functional_call(_model, parameters, xs))
  return fn


def make_functional_fwd_vector_xs(_model, parametersubset, output_func=None):
  if output_func == None: output_func=identity_func
  def fn(parameters, xs):
    paramdict = vector_to_parameterdict(parameters, parametersubset=parametersubset)
    return output_func(functional_call(_model, parameters, xs))
  return fn


def make_functional_fwd(_model, xs, output_func=None):
    if output_func == None: output_func=identity_func
    def fn(parameters):
        return output_func(functional_call(_model, parameters, (xs.unsqueeze(0),)).squeeze(0))
    return fn

def make_functional_fwd_vector(_model, xs, parametersubset, output_func=None):
    if output_func == None: output_func=identity_func
    def fn(parameters):
        paramdict = vector_to_parameterdict(parameters, parametersubset=parametersubset)
        return output_func(functional_call(_model, paramdict, (xs.unsqueeze(0),)).squeeze(0))
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

def functional_loss_for_vmap(model_func, parametersubset, loss_func, xs, ys, prior_loss=None):
    # Returns a function that takes parameters, data, and target and returns the loss
    def fn(parameters):
        param_dict = vector_to_parameterdict(parameters, parametersubset)
        pred = model_func(param_dict, xs)
        loss = loss_func(pred, ys) 
        return loss + prior_loss(parameters) if prior_loss is not None else loss
    return fn



@contextmanager
def torch_seed(seed):
    """
    A context manager to temporarily set the random seed in PyTorch.
    
    Args:
        seed (int): The seed value to use within the context.
    """
    # Save the current random state
    random_state = torch.get_rng_state()
    try:
        torch.manual_seed(seed)
        yield
    finally:
        # Restore the previous random state
        torch.set_rng_state(random_state)




def setup_logger(base_directory, file_logging=True):
    # set up logging
    if file_logging:
        
        logger = logging.getLogger("my-logger")
        # removing old handlers
        logger.handlers.clear()
        
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        os.makedirs(f"{base_directory}/logs/", exist_ok=True)
        handler_file = logging.FileHandler(f"{base_directory}/logs/log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log", mode='w') # and log to file
        print(f"Logging to {handler_file}")
        handler_file.setLevel(logging.DEBUG)
        handler_file.setFormatter(formatter)
        logger.addHandler(handler_file)

        handler_stream = logging.StreamHandler()
        handler_stream.setLevel(logging.DEBUG)
        handler_stream.setFormatter(formatter)
        logger.addHandler(handler_stream)
        logger_info = logger.info

    else:
        logger_info = print
    return logger_info