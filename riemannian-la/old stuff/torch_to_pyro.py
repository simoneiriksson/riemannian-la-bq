import torch
import pyro
import pyro.distributions as dist

def pyro_model_from_pytorch(pytorch_model, priors=None):
    """
    Convert an arbitrary PyTorch model into a Pyro model with priors over its parameters.
    
    Parameters:
    - pytorch_model: a PyTorch model (torch.nn.Module) whose parameters you want to turn into random variables.
    - priors: a dictionary specifying priors for specific parameters. If None, default to Normal(0, 1) priors.
    
    Returns:
    - A Pyro model that can be used with inference algorithms like MCMC.
    """
    # Use default Normal(0, 1) priors if no custom priors are provided
    if priors is None:
        priors = {}
    
    def pyro_model(x, y=None):
        # Iterate over all named parameters in the PyTorch model (weights and biases)
        for name, param in pytorch_model.named_parameters():
            # Define prior distributions for the parameters
            # Use provided prior if exists, else default to Normal(0, 1)
            param_prior = priors.get(name, dist.Normal(0., 1.).expand(param.shape).to_event(param.dim()))
            
            # Sample from the prior distribution and replace the parameter
            sampled_param = pyro.sample(name, param_prior)
            
            # Manually set the sampled parameter back into the PyTorch model
            setattr(pytorch_model, name.replace('.', '_'), torch.nn.Parameter(sampled_param))
        
        # Make predictions using the PyTorch model
        y_pred = pytorch_model(x)
        
        # Likelihood for observed data
        with pyro.plate('data', x.shape[0]):
            pyro.sample('obs', dist.Normal(y_pred, 0.1), obs=y)
    
    return pyro_model
