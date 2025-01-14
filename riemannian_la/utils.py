import torch

def tensify(variable):
    if isinstance(variable, torch.Tensor):
        return variable
    elif variable==None:
        return None
    else: 
        return torch.tensor(variable, dtype=torch.float32)


def NegLogLik_regression(target_sigma=1.0):
    def fn(pred, target):
        #loss = (pred - target).pow(2).sum()/(2 * target_sigma**2)
        loss = torch.nn.MSELoss(reduction="sum")(pred, target)/(2 * tensify(target_sigma)**2)
        return loss
    return fn

def iid_gaussian_prior(prior_sigma=1.0):
    if prior_sigma == 0:  # if prior_sigma is zero, return a function that returns zero - that is: no regularization
        def fn(parameters):
            return torch.tensor(0.0)
        return
    def fn(parameters):
        return parameters.pow(2).sum()/(2 * prior_sigma**2)
    return fn

NegLogLik_classification = lambda pred, target: torch.nn.CrossEntropyLoss(reduction="sum")(pred, target)  # should there be a multiplier here?