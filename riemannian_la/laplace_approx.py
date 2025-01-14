import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.distributions.multivariate_normal import _precision_to_scale_tril
from utils import tensify
from GGN_hessian import GGN_hessian_from_loader
from hessian import hessian_from_model_loss_and_data, hessian_dict_to_matrix, hessian_from_loader, hessian_from_func
from riemannian_la.utils import NegLogLik_regression, NegLogLik_classification, iid_gaussian_prior

class Laplace():
    def __init__(self, model, parametersubset=None, dataloader=None, 
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

        assert (prior_logprob is None) ^ (prior_sigma is not None), "Either prior_logprob or prior_sigma, but not both must be specified"

        if prior_sigma is not None:  # assume Gaussian iid prior if prior_sigma is specified
            self.prior_logprob = iid_gaussian_prior(prior_sigma=prior_sigma)

        assert (target_sigma is not None) and  (loss_fn is None), "Can't specify both target_sigma and loss_fn at the same time"

        if loss_fn is None and target_sigma is not None:  # assume regression
            self.loss_fn = NegLogLik_regression(target_sigma=target_sigma)

        if loss_fn is None and target_sigma is None:  # assume classification
            self.loss_fn = NegLogLik_classification()
        
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
            self.gnn_pre_hessian = hessian_from_loader(self.model, loss_fn=self.loss_fn, xs=xs, ys=ys)

        elif fitting_type == "GGN":
            self.gnn_hessian = GGN_hessian_from_loader(self.model, 
                                        dataloader=dataloader, 
                                        loss_fn=self.loss_fn,
                                        target_sigma=self.target_sigma,
                                        parametersubset=self.parametersubset, 
                                        device=self.device
                                        ).detach()
        if self.prior_sigma is not None:
            self.regularization = torch.eye(self.num_params, device=self.device) / self.prior_sigma**2
        else:
            H = hessian_from_func(self.prior_logprob, self.mean)
            self.regularization = -H
        
        self.precision = self.gnn_pre_hessian + self.regularization
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

    def predictive_posterior_samples(self, xs=None):
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet")

        if not hasattr(self, "posterior_samples"):
            self.make_posterior_sample(self.n_posterior_samples)
        
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
