import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import register_parametrization

class AffineReparametrization(nn.Module):
    def __init__(self, w, b, start_param, end_param, original_shape, reparam_weights):
        super().__init__()
        self.w = w[start_param:end_param]
        self.b = b[start_param:end_param]
        self.start_param = start_param
        self.end_param = end_param
        self.original_shape = original_shape
        self.reparam_weights = reparam_weights
        
    def forward(self, x):
        #print("forward")
        # Apply the affine transformation to the specified slice
        transformed_slice = (self.w @ self.reparam_weights) + self.b 
        #print(f"{transformed_slice.shape = }")
        return transformed_slice.view(self.original_shape)

class AffineSubspaceModel(nn.Module):
    def __init__(self, model_class, sgd_trace, rank=2, model_kargs={}, device="cpu"):
        super().__init__()
        self.device = device
        self.sgd_trace = sgd_trace
        self.rank = rank
        self.model_class = model_class
        self.base_model = model_class(**model_kargs).to(device)

    def fit(self):
        sgd_trace_matrix = torch.row_stack(self.sgd_trace).detach()
        sgd_trace_mean = sgd_trace_matrix.mean(dim=0)
        U, S, V = (sgd_trace_matrix-sgd_trace_mean.unsqueeze(0)).svd()
        subspace_span = V[:, :self.rank]
        subspace_bias = sgd_trace_mean
        self.reparam_weights = nn.Parameter(torch.zeros(self.rank, device=self.device))
        param_dicts = []
        verbose = True

        param_tuples = []
        total_params = 0
        for name, module in self.base_model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                numel = param.numel()
                original_shape = param.shape
                param_tuples.append((module, param_name, total_params, total_params + numel, original_shape))
                total_params += numel

        # Apply reparameterization after collecting all the parameters
        for module, param_name, start_param, end_param, original_shape in param_tuples:
            reparam = AffineReparametrization(subspace_span, subspace_bias, start_param, end_param, original_shape, self.reparam_weights)
            register_parametrization(module, param_name, reparam)
        
    def forward(self, x):
        return self.base_model(x)
