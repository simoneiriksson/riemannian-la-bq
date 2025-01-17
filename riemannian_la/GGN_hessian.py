from torch.func import grad, jvp, vjp, hessian, jacfwd, jacrev, vmap, functional_call
import torch
from utils import tensify, make_functional_fwd_xs

# This function turns a dictionary of gradients into a single vector
def grad_dict_to_vector(grad_dict, verbose=False, output_dims=0, device="cpu"):
    N = list(grad_dict.values())[0].shape[0]
    output_shape = list(list(grad_dict.values())[0].shape[1:output_dims+1])
    grad_size = int(sum([torch.prod(torch.tensor(grad_dict[key][0].shape[output_dims:])) for key in grad_dict.keys()]))
    
    grad_vector = torch.zeros((N, *output_shape, grad_size), device=device)
    index = 0
    for name, value in grad_dict.items():
        numel = torch.prod(torch.tensor(value.shape[1+output_dims:]))
        flatten_grad = torch.flatten(value, start_dim=1+output_dims)
        grad_vector[..., index:index+numel] = flatten_grad
        index += numel
    return grad_vector

def GGN_hessian(model, xs, ys, loss_fn=None, target_sigma=None, parametersubset=None):
    device = xs.device
    if parametersubset is None:
        parametersubset = dict(model.named_parameters())
    model_func = make_functional_fwd_xs(model)  # create functional forward
    jac_fn = jacrev(model_func, argnums=0)  # create jacobian function
    mapped_jac_fn = vmap(jac_fn, in_dims=(None, 0))  # vectorize jacobian function
    
    jacobian_dict = mapped_jac_fn(parametersubset, xs)  # compute jacobian
    jacobian_matrix = grad_dict_to_vector(jacobian_dict, output_dims=1, device=device)  # turn jacobian into matrix
    if loss_fn == "classification":
        pred = model(xs).softmax(dim=-1)
        loss_hessian_matrix = torch.diag_embed(pred) - torch.einsum('nk, nj -> nkj', pred, pred)
        J_H = torch.einsum('nkd, nki -> ndi', jacobian_matrix, loss_hessian_matrix).detach()
        J_H_J = torch.einsum('ndk, nkD -> ndD', J_H, jacobian_matrix).sum(dim=0).detach()

    elif loss_fn == "regression":
        H_factor = 1/tensify(target_sigma).to(device)**2
        loss_hessian_matrix = torch.eye(jacobian_matrix.shape[1], device = jacobian_matrix.device).unsqueeze(0).repeat(jacobian_matrix.shape[0], 1, 1) * H_factor
        J_H = torch.einsum('nkd, nki -> ndi', jacobian_matrix, loss_hessian_matrix)
        J_H_J = torch.einsum('ndk, nkD -> ndD', J_H, jacobian_matrix).sum(dim=0)

    else: 
        # this works at least for loss_fn = torch.nn.CrossEntropyLoss
        pred = model(xs)
        hessian_fn = hessian(loss_fn, argnums=0)  # create hessian function
        mapped_hessian_fn = vmap(hessian_fn)  # vectorize hessian function
        loss_hessian_matrix = mapped_hessian_fn(pred, ys)
        if pred.dim() == 1: # Test if this actually works
            J_H = torch.einsum('nd, n -> nd', jacobian_matrix, loss_hessian_matrix)
            J_H_J = torch.einsum('nd, nD -> dD', J_H, jacobian_matrix)
        elif pred.dim() == 2: # Test if this actually works
            J_H = torch.einsum('nkd, nki -> ndi', jacobian_matrix, loss_hessian_matrix)
            J_H_J = torch.einsum('ndk, nkD -> ndD', J_H, jacobian_matrix).sum(dim=0)
    return J_H_J

def GGN_hessian_from_loader(model, dataloader = None, loss_fn=None, target_sigma=None, parametersubset=None, device="cpu"):
    if parametersubset is None:
        parametersubset = dict(model.named_parameters())
    num_params =  sum([p.numel() for p in parametersubset.values()])
    ggn_hessian = torch.zeros((num_params, num_params), device=device)
    for i, (x, y) in enumerate(dataloader):  # loop over batches
        x_batch = x.to(device)
        y_batch = y.to(device)
        ggn_hessian += GGN_hessian(model, x_batch, y_batch, parametersubset=parametersubset, target_sigma=target_sigma, loss_fn=loss_fn).detach().clone()
    return ggn_hessian