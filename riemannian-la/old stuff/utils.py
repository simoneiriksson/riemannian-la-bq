import numpy as np
import torch
# Kindly copied from https://github.com/wjmaddox/drbayes/blob/master/subspace_inference/utils.py
def set_weights(model, vector, device=None):
    offset = 0
    for param in model.parameters():
        param.data.copy_(vector[offset:offset + param.numel()].view(param.size()).to(device))
        offset += param.numel()

# Kindly copied from https://github.com/wjmaddox/drbayes/blob/master/subspace_inference/utils.py
def set_weights_old(params, w, device):	
    offset = 0
    for module, name, shape in params:
        size = np.prod(shape)	       
        value = w[offset:offset + size]
        setattr(module, name, value.view(shape).to(device))	
        offset += size

# Kindly copied from https://github.com/wjmaddox/drbayes/blob/master/subspace_inference/utils.py
def extract_parameters(model):
    params = []
    for module in model.modules():
        for name in list(module._parameters.keys()):
            if module._parameters[name] is None:
                continue
            param = module._parameters[name]
            params.append((module, name, param.size()))
            module._parameters.pop(name)
    return params

# Function that loop over a list of param_names and return the indices of the parameters in the flattened vector
def get_parameter_vector_indices(model, param_names):
    """
    Returns the index range of a specified parameter in the flattened parameter vector.

    Args:
        model (torch.nn.Module): The model containing parameters.
        param_name (str): The name of the parameter for which to find indices.

    Returns:
        tuple: A tuple (start_index, end_index) representing the indices
               of the specified parameter in the flattened vector, or None if not found.
    """
    start_idx = 0
    indices = []
    for name, param in model.named_parameters():
        num_param_elements = param.numel()  # Total elements in this parameter tensor
        end_idx = start_idx + num_param_elements  # End index is start + size of this parameter

        if name in param_names:
            indices.append(torch.arange(start_idx, end_idx))  # Return indices if parameter name matches

        start_idx = end_idx  # Update start index for next parameter

    return indices  # Return None if the parameter name is not found