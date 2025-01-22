import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.distributions.multivariate_normal import _precision_to_scale_tril
from utils import tensify, loss_func_from_target_sigma, make_functional_fwd_xs, vector_to_parameterdict
from GGN_hessian import GGN_hessian_from_loader
from hessian import hessian_from_model_loss_and_data, hessian_dict_to_matrix, hessian_from_loader, hessian_from_func
from riemannian_la.utils import NegLogLik_regression, NegLogLik_classification, iid_gaussian_prior
from torch.func import grad, jvp, vjp, hessian, jacfwd, jacrev, vmap, functional_call



class Laplace():
    def __init__(self, model, parametersubset=None, dataloader=None, xs=None, ys=None,
                 prior_sigma=None, prior_logprob=None, target_sigma=None, loss_fn=None, device="cpu", verbose=False,
                 n_posterior_samples=1000
                 ):

        self.model = model
        self.parametersubset = parametersubset
        self.device = device
        self.verbose = verbose
        self.is_fitted = False
        self.n_posterior_samples = n_posterior_samples
        self.dataloader = dataloader

        self.prior_sigma = prior_sigma
        self.prior_logprob = prior_logprob
        self.target_sigma = target_sigma
        self.loss_fn = loss_fn
        self.xs = xs
        self.ys = ys
        if ((dataloader is not None) and (xs is not None)) or ((xs is None) and (dataloader is None)):
            raise ValueError("Either dataloader or xs must be provided")
        if xs is not None:
            self.dataloader = DataLoader(TensorDataset(xs, ys), batch_size=len(xs))
        else: self.dataloader = dataloader

        assert (prior_logprob is None) ^ (prior_sigma is None), "Either prior_logprob or prior_sigma, but not both must be specified"

        # if prior_sigma is not None:  # assume Gaussian iid prior if prior_sigma is specified
        #     self.prior_logprob = iid_gaussian_prior(prior_sigma=prior_sigma)

        assert (target_sigma is None) or (loss_fn is None), "Can't specify both target_sigma and loss_fn at the same time"

        self.loss_fn = loss_func_from_target_sigma(loss_fn, target_sigma)

        if parametersubset is None:
            self.parametersubset = dict(model.named_parameters())
        else:
            self.parametersubset = parametersubset
        
        self.mean = torch.nn.utils.parameters_to_vector(self.parametersubset.values())
        self.num_params = self.mean.numel()

    def fit(self, fitting_type="hessian", xs=None, ys=None):
        _=self.model.eval()
        if xs is not None:
            dataloader = DataLoader(TensorDataset(xs, ys), batch_size=len(xs))
        else: dataloader = self.dataloader

        if fitting_type == "hessian":
            self.hessian = hessian_from_loader(model=self.model, dataloader = dataloader, 
                                               loss_fn=self.loss_fn, parametersubset=None, device=self.device)

        elif fitting_type == "GGN":
            self.hessian = GGN_hessian_from_loader(self.model, 
                                        dataloader=dataloader, 
                                        loss_fn=self.loss_fn,
                                        target_sigma=self.target_sigma,
                                        parametersubset=self.parametersubset, 
                                        device=self.device
                                        ).detach()
        if self.prior_sigma is not None:
            if self.prior_sigma == 0:
                self.regularization = torch.zeros_like(self.hessian)
            else:
                self.regularization = torch.eye(self.num_params, device=self.device) / self.prior_sigma**2
        else:
            H = hessian_from_func(self.prior_logprob, self.mean)
            self.regularization = -H
        
        self.precision = self.hessian + self.regularization
        self.is_fitted = True

        self.scale = _precision_to_scale_tril(self.precision)
        self.covariance = self.scale @ self.scale.T
        return self.mean, self.precision

    def make_posterior_sample(self, n_samples=None):
        if n_samples is None:
            n_samples = self.n_posterior_samples
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet")
        eps = torch.randn(n_samples, self.num_params, device=self.device)
        self.posterior_samples = self.mean + eps @ self.scale.T 
        return self.posterior_samples


    def functional_model_bck(self, parameters, xs):
        counter = 0
        for key in self.parametersubset.keys():
            self.parametersubset[key].data = parameters[counter:counter+self.parametersubset[key].numel()].view(self.parametersubset[key].shape)
            counter += self.parametersubset[key].numel()
        return self.model(xs)

    def predictive_posterior_samples(self, xs=None):
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet")

        if not hasattr(self, "posterior_samples"):
            self.make_posterior_sample(self.n_posterior_samples)
        
        functional_model = make_functional_fwd_xs(self.model)
        
        # loop over posterior samples
        for sample_no, posterior_sample in enumerate(self.posterior_samples):
            param_dict = vector_to_parameterdict(posterior_sample, self.parametersubset)
            prediction = functional_model(param_dict, xs)
            if sample_no == 0: 
                predictions = torch.zeros((len(self.posterior_samples), *prediction.shape), device=self.device)
            predictions[sample_no] = prediction
        return predictions

    def predictive_posterior_samples_bck(self, xs=None):
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet")

        if not hasattr(self, "posterior_samples"):
            self.make_posterior_sample(self.n_posterior_samples)

        parametersubset_bck = {key: value.clone() for key, value in self.parametersubset.items()}

        # loop over posterior samples
        for sample_no, posterior_sample in enumerate(self.posterior_samples):
            prediction = self.functional_model(posterior_sample, xs)
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
        return self.predictive_posterior_samples(xs).mean(dim=0)

    def __call__(self, xs):
        return self.forward(xs)

    def __str__(self):
        return f"Laplace approximation for model {self.model} with task type {self.task_type} and parameters {self.parametersubset}"


