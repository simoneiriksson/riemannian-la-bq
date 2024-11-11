
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.func import grad, jvp, vjp, hessian, jacfwd, jacrev, vmap, functional_call

###############################
# LA wrapper function

def make_functional_fwd(_model):
    def fn(parameters, data):
        return functional_call(_model, parameters, (data,)).squeeze(0)
    return fn


def make_loss_func_from_distr(model_func, prior_distribution, likelihood_given_outputs):
    def fn(parameters, data, target):
        param_vector = torch.nn.utils.parameters_to_vector([param for param in parameters.values()]).detach()
        #print(f"{target.shape = }")
        #print(f"{data.shape = }")
        pred = model_func(data, parameters)
        #print(f"{pred.shape = }")
        like = likelihood_given_outputs(pred).log_prob(target).sum()
        reg = prior_distribution.log_prob(param_vector)
        #print(f"{like.shape = }")
        #print(f"{reg.shape = }")
        return -(like+reg)
    return fn

def LA_approximation(model, dataloader = None, xs=None, ys=None, batch_size=16, task_type = None,
                     prior_distribution=None, prior_sigma=None,  
                     likelihood_given_outputs=None, target_sigma=None,
                     parametersubset=None, return_hessian=True, 
                     return_gradient=True, device="cpu", batching=True):
    if (dataloader is not None) and (batch_size is None):
        raise ValueError("If dataloader is provided, batch_size must be provided")
    if ((dataloader is not None) and (xs is not None)) or ((xs is None) and (dataloader is None)):
        raise ValueError("Either dataloader or xs must be provided")
    if ((task_type is None) and (likelihood_given_outputs is None)) or ((task_type is not None) and (likelihood_given_outputs is not None)):
        raise ValueError("Either task_type or likelihood_given_outputs must be provided")
    if ((prior_distribution is None) and (prior_sigma is None)) or ((prior_distribution is not None) and (prior_sigma is not None)):
        raise ValueError("Either prior_distribution or prior_sigma must be provided")
    
    if parametersubset is None:
        parametersubset = model.named_parameters()
        num_params =  sum([p[1].numel() for p in model.named_parameters()])
    else:
        num_params =  sum([p.numel() for p in parametersubset.values()])

    # If no prior_distribution is provided, create one, from prior_sigma
    if prior_distribution is None:
        prior_distribution = torch.distributions.MultivariateNormal(torch.zeros(num_params), torch.eye(num_params)*prior_sigma*num_params)

    # If no likelihood_given_outputs is provided, create one, from task_type
    if likelihood_given_outputs is None:
        if task_type == "regression":
            likelihood_given_outputs = lambda x: torch.distributions.Normal(loc=x, scale=target_sigma) 
        elif task_type == "classification":
            likelihood_given_outputs = lambda x: torch.distributions.Categorical(logits=x)
        else:
            raise ValueError("task_type not recognized")

    model_func = make_functional_fwd(model)  # functional forward
    params_used = dict(parametersubset)  # parameters used in the loss function
    loss_func = make_loss_func_from_distr(model_func, prior_distribution, likelihood_given_outputs)  # loss function
    gradient_fn = grad(loss_func, argnums=0)  # gradient function
    return_vals = []
    if batching:
        # If no dataloader is provided, create one
        if dataloader is None:
            dataloader = DataLoader(TensorDataset(xs, ys), batch_size=batch_size)
        gradients = {}
        hessians = {}
        for i, (x, y) in enumerate(dataloader):  # loop over batches
            x, y = x.to(device), y.to(device)
            if return_gradient:
                #print(f"x.shape = {x.shape}")
                #print(f"y.shape = {y.shape}")
                gradient = gradient_fn(params_used, x, y)
                for key, value in gradient.items():
                    if key in gradients:
                        gradients[key] += value.detach()
                    else:
                        gradients[key] = value.detach()
            if return_hessian:
                hess_fn = hessian(loss_func, argnums=0)
                hess = hess_fn(params_used, x, y)
                for key1, value1 in hess.items():
                    if key1 not in hessians:
                        hessians[key1] = {}
                    for key2, value2 in value1.items():
                        if key2 in hessians[key1]:
                            hessians[key1][key2] += value2.detach()
                        else:
                            hessians[key1][key2] = value2.detach()
        if return_gradient: 
            return_vals.append(gradients)
        if return_hessian:
            return_vals.append(hessians)
    else:
        if return_gradient:
            gradient = gradient_fn(params_used, xs, ys)
            return_vals.append(gradient)
        if return_hessian:
            hess_fn = hessian(loss_func, argnums=0)
            hess = hess_fn(params_used, xs, ys)
            return_vals.append(hess)
    return return_vals



def hessian_dict_to_matrix(hess_dict, verbose=False):
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

    hess_matrix = torch.zeros((hess_size, hess_size))
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




def grad_dict_to_vector(grad_dict, verbose=False):
    N = list(grad_dict.values())[0].shape[0]
    grad_size = sum([len(torch.flatten(grad_dict[key][0])) for key in grad_dict.keys()])
    grad_vector = torch.zeros((N, grad_size))
    index = 0
    for name, value in grad_dict.items():
        numel = torch.prod(torch.tensor(value.shape[1:]))
        flatten_grad = torch.flatten(value, start_dim=1)
        grad_vector[:,index:index+numel] = flatten_grad
        index += numel
    return grad_vector


def GNN_hessian(model, x_train, parametersubset=None):
    if parametersubset is None:
        parametersubset = dict(model.named_parameters())
    #print(f"{parametersubset = }")
    model_func = make_functional_fwd(model)
    ft_comute_grad = grad(model_func, argnums=0)
    mapped_ft = vmap(ft_comute_grad, in_dims=(None, 0))
    grad_dict = mapped_ft(parametersubset, x_train)
    grads = grad_dict_to_vector(grad_dict)
    ggn_hessian = grads.permute(1,0) @ grads
    return ggn_hessian


# def GNN_hessian(model, x_train, parametersubset=None):
#     model_func = make_functional_fwd(model)
#     num_params = sum([p[1].numel() for p in model.named_parameters()])
#     ggn_hessian = torch.zeros((num_params, num_params))
#     ft_comute_grad = grad(model_func, argnums=0)
#     mapped_ft = vmap(ft_comute_grad, in_dims=(None, 0))
#     params = dict(model.named_parameters())
#     grad_dict = mapped_ft(params, x_train)
#     grads = grad_dict_to_vector(grad_dict)
#     ggn_hessian += grads.permute(1,0) @ grads

#     return ggn_hessian

def GNN_posterior_precision(model, task_type, parametersubset=None, dataloader = None, batch_size=16, xs=None, prior_sigma=None, target_sigma=None, device="cpu", batching=True):
    if ((dataloader is not None) and (xs is not None)) or ((xs is None) and (dataloader is None)):
        raise ValueError("Either dataloader or xs must be provided")
    if parametersubset is None:
        parametersubset = dict(model.named_parameters())
    #    num_params =  sum([p[1].numel() for p in model.named_parameters()])
    #else:
    num_params =  sum([p.numel() for p in parametersubset.values()])

    if task_type == "regression": 
        # Here \nabla^2 log p(y|x, \theta) = -1/\sigma^2 
        # since p(y|x, \theta) = Normal(y|f(x, \theta), \sigma^2)
        H_factor = 1/target_sigma**2
    elif task_type == "classification":
        # Here we use cross entropy, ln p(y|x, \theta) = ln Categorical(y|f(x, \theta)) = ln \prod_i (exp(f_i)/\sum_j exp(f_j))^{y_i} = \sum_i y_i f_i - \sum_i ln \sum_j exp(f_j)
        # and \nabla^2 log p(y|x, \theta) = 1
        H_factor = -1
    else:
        raise ValueError("task_type not recognized")
    if prior_sigma is None:
        regularization = torch.zero(num_params)
    else:
        regularization = torch.eye(num_params)*1/prior_sigma**2
    if batching:
        ggn_hessian = torch.zeros((num_params, num_params))
        # If no dataloader is provided, create one
        if dataloader is None:
            dataloader = DataLoader(TensorDataset(xs, xs), batch_size=batch_size)
        for i, (x, y) in enumerate(dataloader):  # loop over batches
            x_batch, y_batch = x.to(device), y.to(device)
            ggn_hessian += GNN_hessian(model, x_batch, parametersubset=parametersubset)
    else:
        ggn_hessian = GNN_hessian(model, xs, parametersubset=parametersubset)
    print(f"{ggn_hessian = } before regularization and factoring")
    print(f"{regularization = }")
    print(f"{H_factor = }")
    ggn_hessian *= H_factor
    ggn_hessian += regularization
    return ggn_hessian
