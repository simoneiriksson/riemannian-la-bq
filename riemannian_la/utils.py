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

def NegLogLik_classification():
    def fn(pred, target):
        return torch.nn.CrossEntropyLoss(reduction="sum")(pred, target)
    return fn

def loss_func_from_target_sigma(loss_fn, target_sigma):
    if loss_fn is None and target_sigma is not None:  # assume regression
        loss_fn = NegLogLik_regression(target_sigma=target_sigma)

    if loss_fn is None and target_sigma is None:  # assume classification
        loss_fn = NegLogLik_classification()
    return loss_fn
