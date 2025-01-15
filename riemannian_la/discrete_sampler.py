import torch
from utils import functional_loss_for_vmap, make_functional_fwd_xs
from torch.func import vmap

class discrete_model_sampler:
    def __init__(self, model, loss_fn, xs, ys, limits, n_mesh=100, normalize_weights=True, parametersubset=None):
        if parametersubset is None:
            self.parametersubset = dict(model.named_parameters())
        else:
            self.parametersubset = parametersubset
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
        self.tile_size = self.size / self.n_mesh**self.dims
        self.normalize_weights = normalize_weights
        self.mesh_vals = [torch.linspace(l[0]+self.span[i]/self.n_mesh, l[1]-self.span[i]/self.n_mesh, self.n_mesh) for i, l in enumerate(self.limits)]

    def samples_and_weights(self):
        # make meshgrid
        meshgrid = torch.meshgrid(self.mesh_vals)
        meshgrid = torch.stack(meshgrid, dim=-1)
        samples = meshgrid.view(-1, len(self.limits))
        # weights here should be equal to the likelihood of the data given the parameters
        model_functional = make_functional_fwd_xs(self.model)  # get me the functional version of the model
        loss_functional = functional_loss_for_vmap(model_functional, self.parametersubset, self.loss_fn, xs, ys)
        weights = vmap(loss_functional)(samples)* self.tile_size
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
        self.mesh_vals = [torch.linspace(l[0]+self.span[i]/self.n_mesh, l[1]-self.span[i]/self.n_mesh, self.n_mesh) for i, l in enumerate(self.limits)]

    def samples_and_weights(self):
        # make meshgrid
        meshgrid = torch.meshgrid(self.mesh_vals)
        meshgrid = torch.stack(meshgrid, dim=-1)
        samples = meshgrid.view(-1, len(self.limits))
        weights = vmap(self.func)(samples).view(-1) * self.tile_size
        if self.normalize_weights:
            weights = weights / (weights.sum())  # Does this make sense at all?
        return samples, weights