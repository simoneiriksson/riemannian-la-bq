
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.func import grad, jvp, vjp, hessian, jacfwd, jacrev, vmap, functional_call


###############################
# LA wrapper function

def make_functional_fwd(_model):
    def fn(data, parameters):
        return functional_call(_model, parameters, (data,))
    return fn

def make_loss_func_from_distr(model_func, prior_distribution, likelihood_given_outputs):
    def fn(parameters, data, target):
        param_vector = torch.nn.utils.parameters_to_vector([param for param in parameters.values()]).detach()
        like = likelihood_given_outputs(model_func(data, parameters)).log_prob(target).sum(dim=0)
        reg = prior_distribution.log_prob(param_vector)
        return -(like+reg)[0]
    return fn

def LA_approximation(model, dataloader = None, xs=None, ys=None, batch_size=16, task_type = None,
                     prior_distribution=None, prior_sigma=None,  
                     likelihood_given_outputs=None, target_log_sigma=None,
                     parametersubset=None, return_hessian=True, 
                     return_gradient=True):
    if (dataloader is not None) and (batch_size is None):
        raise ValueError("If dataloader is provided, batch_size must be provided")
    if ((dataloader is not None) and (xs is not None)) or ((xs is None) and (dataloader is None)):
        raise ValueError("Either dataloader or xs must be provided")
    if ((task_type is None) and (likelihood_given_outputs is None)) or ((task_type is not None) and (likelihood_given_outputs is not None)):
        raise ValueError("Either task_type or likelihood_given_outputs must be provided")
    if ((prior_distribution is None) and (prior_sigma is None)) or ((prior_distribution is not None) and (prior_sigma is not None)):
        raise ValueError("Either prior_distribution or prior_sigma must be provided")
   
    # If no dataloader is provided, create one
    if dataloader is None:
        dataloader = DataLoader(TensorDataset(xs, ys), batch_size=batch_size)
    
    if parametersubset is None:
        parametersubset = model.named_parameters()
        num_params =  sum([p[1].numel() for p in model.named_parameters()])
    else:
        num_params =  sum([p.numel() for p in parametersubset.values()])

    # If no prior_distribution is provided, create one, from prior_sigma
    if prior_distribution is None:
        prior_distribution = torch.distributions.MultivariateNormal(torch.zeros(num_params), torch.eye(num_params)*prior_sigma.exp()*num_params)

    # If no likelihood_given_outputs is provided, create one, from task_type
    if likelihood_given_outputs is None:
        if task_type == "regression":
            likelihood_given_outputs = lambda x: torch.distributions.Normal(loc=x, scale=target_log_sigma.exp()) 
        elif task_type == "classification":
            likelihood_given_outputs = lambda x: torch.distributions.Categorical(logits=x)
        else:
            raise ValueError("task_type not recognized")
    
    model_func = make_functional_fwd(model)  # functional forward
    params_used = dict(parametersubset)  # parameters used in the loss function
    loss_func = make_loss_func_from_distr(model_func, prior_distribution, likelihood_given_outputs)  # loss function
    gradient_fn = grad(loss_func, argnums=0)  # gradient function
    return_vals = []
    
    if return_gradient: 
        gradient = gradient_fn(params_used, xs, ys)
        return_vals.append(gradient)
    if return_hessian:
        hess_fn = hessian(loss_func, argnums=0)
        hess = hess_fn(params_used, xs, ys)
        return_vals.append(hess)
    return return_vals
