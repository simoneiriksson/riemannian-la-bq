import torch
from utils import functional_loss_for_vmap, make_functional_fwd_xs, loss_func_from_target_sigma, iid_gaussian_prior
from torch.func import vmap
from torch.utils.data import DataLoader, TensorDataset

class discrete_model_sampler:
    def __init__(self, model=None, dataloader=None, xs=None, ys=None, limits=None, n_mesh=100, normalize_weights=True, parametersubset=None, 
                 prior_sigma=None, prior_logprob=None, target_sigma=None, loss_fn=None):
        if parametersubset is None:
            self.parametersubset = dict(model.named_parameters())
        else:
            self.parametersubset = parametersubset
        self.numparams = torch.nn.utils.parameters_to_vector(self.parametersubset.values()).numel()

        self.model = model
        self.loss_fn = loss_fn
        self.xs = xs
        self.ys = ys
        self.samples = None
        self.weights = None
        self.limits = limits
        self.n_mesh = n_mesh
        self.discrete_sampler = True
        self.span = [abs(l[1] - l[0])  for l in self.limits]
        self.size = torch.tensor(self.span).prod()
        self.dims = len(self.limits)
        if self.dims != self.numparams:
            raise ValueError("The number of limits must be equal to the number of parameters to be integrated over")
        
        self.tile_size = self.size / self.n_mesh**self.dims
        self.normalize_weights = normalize_weights
        self.mesh_vals = [torch.linspace(l[0] + 0.5*self.span[i]/self.n_mesh, l[1] - 0.5*self.span[i]/self.n_mesh, self.n_mesh) for i, l in enumerate(self.limits)]
        if ((dataloader is not None) and (xs is not None)) or ((xs is None) and (dataloader is None)):
            raise ValueError("Either dataloader or xs must be provided")
        if xs is not None:
            self.dataloader = DataLoader(TensorDataset(xs, ys), batch_size=len(xs))
        else: self.dataloader = dataloader

        assert (prior_logprob is None) ^ (prior_sigma is None), "Either prior_logprob or prior_sigma, but not both must be specified"
        assert (target_sigma is None) or (loss_fn is None), "Can't specify both target_sigma and loss_fn at the same time"
        self.prior_sigma = prior_sigma
        self.prior_logprob = prior_logprob
        self.loss_fn = loss_func_from_target_sigma(loss_fn, target_sigma)
        if self.prior_sigma is not None:
            if self.prior_sigma == 0:
                self.prior_logprob = lambda parameters: torch.tensor(0.0)
            else:
                self.prior_logprob = lambda parameters: iid_gaussian_prior(prior_sigma=self.prior_sigma)(parameters)

        if parametersubset is None:
            self.parametersubset = dict(model.named_parameters())
        else:
            self.parametersubset = parametersubset

    def samples_and_weights(self):
        # make meshgrid
        meshgrid = torch.meshgrid(self.mesh_vals)
        meshgrid = torch.stack(meshgrid, dim=-1)
        samples = meshgrid.view(-1, len(self.limits))
        # weights here should be equal to the likelihood of the data given the parameters
        model_functional = make_functional_fwd_xs(self.model)  # get me the functional version of the model
        loss_functional = functional_loss_for_vmap(model_functional, self.parametersubset, self.loss_fn, 
                                                   self.xs, self.ys, prior_logprob=self.prior_logprob)
        weights = (-vmap(loss_functional)(samples)).exp()
        weights = weights * self.tile_size
        if self.normalize_weights:
            weights = weights / (weights.sum())  # Does this make sense at all?
        return samples, weights

class discrete_function_sampler:
    def __init__(self, func, limits, n_mesh=100, normalize_weights=False):
        self.func = func
        self.samples = None
        self.weights = None
        self.limits = limits
        self.n_mesh = n_mesh
        self.discrete_sampler = True
        self.span = [abs(l[1] - l[0])  for l in self.limits]
        self.size = torch.tensor(self.span).prod()
        self.dims = len(self.limits)
        self.tile_size = self.size / self.n_mesh**self.dims
        self.normalize_weights = normalize_weights
        self.mesh_vals = [torch.linspace(l[0] + .5*self.span[i]/self.n_mesh, l[1] - .5*self.span[i]/self.n_mesh, self.n_mesh) for i, l in enumerate(self.limits)]

    def samples_and_weights(self):
        # make meshgrid
        meshgrid = torch.meshgrid(self.mesh_vals)
        meshgrid = torch.stack(meshgrid, dim=-1)
        samples = meshgrid.view(-1, len(self.limits))
        weights = vmap(self.func)(samples).view(-1) * self.tile_size
        if self.normalize_weights:
            weights = weights / (weights.sum())  # Does this make sense at all?
        return samples, weights