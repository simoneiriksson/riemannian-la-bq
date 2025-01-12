
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.func import grad, jvp, vjp, hessian, jacfwd, jacrev, vmap, functional_call
from utils import tensify
from torch.distributions.multivariate_normal import _precision_to_scale_tril

###############################
# LA wrapper function

def make_functional_fwd(_model):
    def fn(parameters, data):
        return functional_call(_model, parameters, (data.unsqueeze(0),)).squeeze(0)
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





# This function turns a dictionary of gradients into a single vector
def grad_dict_to_vector(grad_dict, verbose=False, output_dims=0, device="cpu"):
    N = list(grad_dict.values())[0].shape[0]
    output_shape = list(list(grad_dict.values())[0].shape[1:output_dims+1])
    grad_size = sum([torch.prod(torch.tensor(grad_dict[key][0].shape[output_dims:])) for key in grad_dict.keys()])
   
    grad_vector = torch.zeros((N, *output_shape, grad_size), device=device)
    index = 0
    for name, value in grad_dict.items():
        numel = torch.prod(torch.tensor(value.shape[1+output_dims:]))
        flatten_grad = torch.flatten(value, start_dim=1+output_dims)
        grad_vector[..., index:index+numel] = flatten_grad
        index += numel
    return grad_vector


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

# this function returns $\nabla_\theta f(x, \theta) \nabla^2 loss(pred, target) \nabla_\theta f(x, \theta)^T$
def GGN_hessian(model, xs, ys, loss_fn=None, target_sigma=None, output_dims=0, parametersubset=None):
    device = xs.device
    if parametersubset is None:
        parametersubset = dict(model.named_parameters())
    model_func = make_functional_fwd(model)  # create functional forward
    jac_fn = jacrev(model_func, argnums=0)  # create jacobian function
    mapped_jac_fn = vmap(jac_fn, in_dims=(None, 0))  # vectorize jacobian function
    jacobian_dict = mapped_jac_fn(parametersubset, xs)  # compute jacobian
    jacobian_matrix = grad_dict_to_vector(jacobian_dict, output_dims=1, device=device)  # turn jacobian into matrix
    #jacobian_matrix_perm = jacobian_matrix.permute([*list(range(1, 1+output_dims)), 0, -1])  # permute jacobian matrix
    #ggn_hessian = torch.bmm(jacobian_matrix_perm.transpose(-1, -2), jacobian_matrix_perm)  # compute hessian
    if loss_fn == "classification":
        pred = model(xs).softmax(dim=-1)
        loss_hessian_matrix = torch.diag_embed(pred) - torch.einsum('nk, nj -> nkj', pred, pred)
        #J_H_J = torch.einsum('nkd, nkk, nkD-> dD', jacobian_matrix, loss_hessian_matrix, jacobian_matrix)
        #J_H = torch.einsum('nkd, nkk -> ndk', jacobian_matrix, loss_hessian_matrix)
        #J_H_J = torch.einsum('ndD, nkO -> dO', J_H, jacobian_matrix)
        #J_H = torch.einsum('nkd, ndd -> ndk', jacobian_matrix, loss_hessian_matrix)
        #J_H_J = torch.einsum('nd, nkD -> ndD', J_H, jacobian_matrix).sum(dim=0)
        J_H = torch.einsum('nkd, nki -> ndi', jacobian_matrix, loss_hessian_matrix)
        J_H_J = torch.einsum('ndk, nkD -> ndD', J_H, jacobian_matrix).sum(dim=0)

    elif loss_fn == "regression":
        H_factor = 1/tensify(target_sigma)**2
        loss_hessian_matrix = torch.eye(jacobian_matrix.shape[1], device = jacobian_matrix.device).unsqueeze(0).repeat(jacobian_matrix.shape[0], 1, 1) * H_factor
        #print(f"{jacobian_matrix.shape = }, {loss_hessian_matrix.shape = }")
        # k=output dimension, D,d=parameter dimension, n=batch dimension
        #J_H = torch.einsum('nkd, nkk -> ndk', jacobian_matrix, loss_hessian_matrix)
        #J_H_J = torch.einsum('ndD, nkD -> ndD', J_H, jacobian_matrix).sum(dim=0)
        J_H = torch.einsum('nkd, nki -> ndi', jacobian_matrix, loss_hessian_matrix)
        J_H_J = torch.einsum('ndk, nkD -> ndD', J_H, jacobian_matrix).sum(dim=0)
        #J_H_J = torch.einsum('nkd, nkk, nkD-> dD', jacobian_matrix, loss_hessian_matrix, jacobian_matrix)
    else: 
        # this works at least for loss_fn = torch.nn.CrossEntropyLoss
        pred = model(xs)
        hessian_fn = hessian(loss_fn, argnums=0)  # create hessian function
        mapped_hessian_fn = vmap(hessian_fn)  # vectorize hessian function
        loss_hessian_matrix = mapped_hessian_fn(pred, ys).squeeze(0)
        if pred.dim == 1: # Test if this actually works
            #J_H_J = torch.einsum('nd, n, nD-> dD', jacobian_matrix, loss_hessian_matrix, jacobian_matrix)
            J_H = torch.einsum('nd, n -> nd', jacobian_matrix, loss_hessian_matrix)
            J_H_J = torch.einsum('nd, nD -> dD', J_H, jacobian_matrix)
        elif pred.dim == 2: # Test if this actually works
            #J_H_J = torch.einsum('nkd, nkk, nkD-> dD', jacobian_matrix, loss_hessian_matrix, jacobian_matrix)
            J_H = torch.einsum('nkd, nki -> ndi', jacobian_matrix, loss_hessian_matrix)
            J_H_J = torch.einsum('ndk, nkD -> ndD', J_H, jacobian_matrix).sum(dim=0)
    return J_H_J

# This function returns the Generalized Gauss-Newturn approximation of the hessian of the model with respect to the parameters, given the input data
# that is, it returns $\nabla_\theta f(x, \theta) \nabla_\theta f(x, \theta)^T * \nabla^2_f(x) loss(y, f(x))$
# but before multiplying the Hessian of the loss and adding the regularization term
def GGN_pre_hessian(model, parametersubset=None, dataloader = None, batch_size=16, xs=None, ys=None,
                    device="cpu", batching=True, verbose=False, target_sigma=None, loss_fn=None):
    if ((dataloader is not None) and (xs is not None)) or ((xs is None) and (dataloader is None)):
        raise ValueError("Either dataloader or xs must be provided")
    if parametersubset is None:
        parametersubset = dict(model.named_parameters())
    num_params =  sum([p.numel() for p in parametersubset.values()])

    if batching:
        ggn_pre_hessian = torch.zeros((num_params, num_params), device=device)
        # If no dataloader is provided, create one
        if dataloader is None:
            dataloader = DataLoader(TensorDataset(xs, xs), batch_size=batch_size)
        for i, (x, y) in enumerate(dataloader):  # loop over batches
            if verbose:
                print(f"\nBatch number {i} out of {len(dataloader)}")
            x_batch = x.to(device)
            y_batch = y.to(device)
            ggn_pre_hessian += GGN_hessian(model, x_batch, y_batch, parametersubset=parametersubset, target_sigma=target_sigma, loss_fn=loss_fn).detach().clone()
    else:
        ggn_pre_hessian = GGN_hessian(model, xs, ys, parametersubset=parametersubset, target_sigma=target_sigma, loss_fn=loss_fn).detach().clone()
    return ggn_pre_hessian

class Laplace():
    def __init__(self, model, task_type, parametersubset=None, dataloader=None, xs=None, ys=None, 
                 batch_size=256, prior_sigma=None, target_sigma=None, device="cpu", verbose=False,
                 n_posterior_samples=1000
                 ):
        self.model = model
        self.task_type = task_type
        self.parametersubset = parametersubset
        self.prior_sigma = tensify(prior_sigma)
        self.device = device
        self.verbose = verbose
        self.target_sigma = tensify(target_sigma)
        self.xs = xs
        self.ys = ys
        self.batch_size = batch_size
        self.is_fitted = False
        self.batching = True
        self.n_posterior_samples = n_posterior_samples
        
        if ((dataloader is not None) and (xs is not None)) or ((xs is None) and (dataloader is None)):
            raise ValueError("Either dataloader or xs must be provided")
        if xs is not None:
            self.dataloader = DataLoader(TensorDataset(xs, ys), batch_size=len(xs))
        else: self.dataloader = dataloader

        if parametersubset is None:
            self.parametersubset = dict(model.named_parameters())
        else:
            self.parametersubset = parametersubset
        self.mean = torch.nn.utils.parameters_to_vector(self.parametersubset.values())
        self.num_params = self.mean.numel()
        
        if prior_sigma is None:
            self.regularization = torch.tensor(0.).to(self.device)
        else:
            self.regularization = 1/(self.prior_sigma**2).to(self.device)
    
    def fit_ggn(self, xs=None, ys=None):
        #_=self.model.eval()
        if xs is not None:
            dataloader = DataLoader(TensorDataset(xs, ys), batch_size=len(xs))
        else: dataloader = self.dataloader
        self.gnn_pre_hessian = GGN_pre_hessian(self.model, 
                                            parametersubset=self.parametersubset, 
                                            dataloader=dataloader, 
                                            batch_size=self.batch_size, 
                                            device=self.device, 
                                            batching=self.batching, verbose=self.verbose, 
                                            loss_fn=self.task_type, target_sigma=self.target_sigma)
        #print(f"{gnn_pre_hessian.device = }, {gnn_pre_hessian.shape = }", flush=True)
        #print(f"{self.regularization.device = }, {self.regularization.shape = }", flush=True)
        #print(f"{self.H_factor.device = }, {self.H_factor.shape = }", flush=True)
        self.gnn_pre_hessian = self.gnn_pre_hessian #/ self.num_params 
            # why this? So that we get same result as in laplace-torch and the analytic solution in the lineaer case.
        num_obs = len(dataloader.dataset)
        self.precision = self.gnn_pre_hessian + self.regularization * torch.eye(self.num_params, device=self.device) 
        #print(f"{self.precision.device = }", flush=True)
        self.is_fitted = True

        # Python dies when i try to invert the precision matrix on the MPS device
        #if self.precision.device.type == "mps":
        #    self.covariance = torch.linalg.inv(self.precision.to("cpu")).to(self.device)
        #else:
        #    self.covariance = torch.linalg.inv(self.precision)
        #self.cholesky = torch.linalg.cholesky(self.covariance)
        self.scale = _precision_to_scale_tril(self.precision)
        self.covariance = self.scale @ self.scale.T
        return self.mean, self.precision

    def make_posterior_sample(self, n_samples=None):
        if n_samples is None:
            n_samples = self.n_posterior_samples
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet")
        # The operator 'aten::linalg_cholesky_ex.L' is not currently implemented for the MPS device
        if self.precision.device.type == "mps":
            covariance = self.covariance.to("cpu")
            mean = self.mean.to("cpu")
        else: 
            covariance = self.covariance
            mean = self.mean
        #print(f"{covariance.device = }")
        #print(f"{mean.device = }")
        self.posterior_samples = torch.distributions.MultivariateNormal(mean, covariance).sample((n_samples,)).to(self.device)
        return self.posterior_samples

    def predictive_posterior(self, xs=None):
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet")

        if not hasattr(self, "posterior_samples"):
            self.make_posterior_sample(self.n_posterior_samples)
        
        # be aware that if you do this over reparametrized parameters, you mess up with the reparametrization
        # I think at least, that the reparametrization will be overwritten below.
        parametersubset_bck = {key: value.clone() for key, value in self.parametersubset.items()}
        
        # loop over posterior samples
        for sample_no, posterior_sample in enumerate(self.posterior_samples):
            #print(f"Making prediction for sample number {sample_no}")
            counter = 0
            # update parametersubset with the posterior sample
            for key in self.parametersubset.keys():
                self.parametersubset[key].data = posterior_sample[counter:counter+self.parametersubset[key].numel()].view(self.parametersubset[key].shape)
                counter += self.parametersubset[key].numel()
            # make prediction
            prediction = self.model(xs)
            # initiate predictions tensor
            if sample_no == 0: 
                predictions = torch.zeros((len(self.posterior_samples), *prediction.shape), device=self.device)
            predictions[sample_no] = prediction
        # reset parametersubset to original values
        counter = 0
        for key in self.parametersubset.keys():
            self.parametersubset[key].data = parametersubset_bck[key]
            counter += self.parametersubset[key].numel()
        return predictions

    def forward(self, xs):
        return self.predictive_posterior(xs).mean(dim=0)

    def __call__(self, xs):
        return self.forward(xs)

    def __str__(self):
        return f"Laplace approximation for model {self.model} with task type {self.task_type} and parameters {self.parametersubset}"