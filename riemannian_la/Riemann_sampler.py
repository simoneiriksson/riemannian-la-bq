import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.distributions.multivariate_normal import _precision_to_scale_tril
from utils import tensify, loss_func_from_target_sigma, make_functional_fwd_xs, vector_to_parameterdict
from GGN_hessian import GGN_hessian_from_loader
from hessian import hessian_from_model_loss_and_data, hessian_dict_to_matrix, hessian_from_loader, hessian_from_func
from riemannian_la.utils import NegLogLik_regression, NegLogLik_classification, iid_gaussian_prior
from torch.func import grad, jvp, vjp, hessian, jacfwd, jacrev, vmap, functional_call
from laplace_approx import Laplace

def functional_loss_for_vmap(model_func, parametersubset, loss_func, xs, ys):
    # Returns a function that takes parameters, data, and target and returns the loss
    def fn(parameters):
        param_dict = vector_to_parameterdict(parameters, parametersubset)
        pred = model_func(param_dict, xs)
        loss = loss_func(pred, ys)
        return loss
    return fn

class Riemann_sampler(Laplace):
    def __init__(self, model, parametersubset=None, dataloader=None, xs=None, ys=None,
                 prior_sigma=None, prior_logprob=None, target_sigma=None, loss_fn=None, device="cpu", verbose=False,
                 n_posterior_samples=1000
                 ):

        super().__init__(model, parametersubset, dataloader, xs, ys, prior_sigma, prior_logprob, target_sigma, loss_fn, device, verbose, n_posterior_samples)

    def fit(self, fitting_type="hessian", xs=None, ys=None):
        super().fit(fitting_type, xs, ys)

    def ode_fun(self, t, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
        theta = state_tensor[:self.MAP.shape[0]]
        v = state_tensor[self.MAP.shape[0]:]
        f_model = make_functional_fwd_xs(self.model)
        f_loss = functional_loss_for_vmap(f_model, self.parametersubset, self.loss_fn, self.xs, self.ys)
        grad_val = grad(f_loss)(theta)
        hess_val = hessian(f_loss)(theta)
        acc = -(grad_val * (1 / (1 + grad_val.norm()**2)) * (v.T @ hess_val @ v)).flatten()
        return torch.cat([v, acc]).to("cpu").detach().numpy()
    