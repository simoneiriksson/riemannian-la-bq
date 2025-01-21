import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.distributions.multivariate_normal import _precision_to_scale_tril
from utils import tensify, loss_func_from_target_sigma, make_functional_fwd_xs, vector_to_parameterdict
from GGN_hessian import GGN_hessian_from_loader
from hessian import hessian_from_model_loss_and_data, hessian_dict_to_matrix, hessian_from_loader, hessian_from_func
from riemannian_la.utils import NegLogLik_regression, NegLogLik_classification, iid_gaussian_prior
from torch.func import grad, jvp, vjp, hessian, jacfwd, jacrev, vmap, functional_call
from laplace_approx import Laplace
from scipy.integrate import solve_ivp
from utils import make_functional_fwd_xs, functional_loss_for_vmap, neglog_loss
from models import functional_banana, Model_from_func
from matplotlib import pyplot as plt
from MCMC_sampler import MCMC_sampler
import torchdiffeq
import torchode

def plot_traj(R_sampler):
    plt.scatter(R_sampler.posterior_samples[:, 0], R_sampler.posterior_samples[:, 1])
    cmap = plt.get_cmap("gist_rainbow")
    for i, sample in enumerate(R_sampler.trajectories):
        color_no = (i*cmap.N)//len(R_sampler.trajectories)
        plt.plot(sample[0], sample[1], marker="x", c=cmap(color_no))
        dx = R_sampler.posterior_samples_la[i][0] - R_sampler.mean[0]
        dy = R_sampler.posterior_samples_la[i][1] - R_sampler.mean[1]
        plt.arrow(R_sampler.mean[0].detach(), R_sampler.mean[1].detach(), dx.detach(), dy.detach(), head_width=0.1, head_length=0.1, color=cmap(color_no))


class Riemann_sampler(Laplace):
    def __init__(self, model, parametersubset=None, dataloader=None, xs=None, ys=None,
                 prior_sigma=None, prior_logprob=None, target_sigma=None, loss_fn=None, device="cpu", verbose=False,
                 n_posterior_samples=1000, rtol=1e-6, atol=1e-6
                 ):

        super().__init__(model, parametersubset, dataloader, xs, ys, prior_sigma, prior_logprob, target_sigma, loss_fn, device, verbose, n_posterior_samples)
        # self.make_posterior_sample_la = super().make_posterior_sample
        self.rtol = rtol
        self.atol = atol

    def make_posterior_sample_la(self, n_samples=None):
        self.posterior_samples_la = super().make_posterior_sample(n_samples)
        #print(f"{self.posterior_samples_la = }")
        self.posterior_samples = None

    def fit(self, fitting_type="hessian", xs=None, ys=None):
        super().fit(fitting_type, xs, ys)
    
    def ode_fun_torch(self, t, state):
        theta = state[:self.num_params]
        v = state[self.num_params:]
        f_model = make_functional_fwd_xs(self.model)
        f_loss = functional_loss_for_vmap(f_model, self.parametersubset, self.loss_fn, self.xs, self.ys, prior_logprob=self.prior_logprob)
        grad_val = grad(f_loss)(theta)
        hess_val = hessian(f_loss)(theta).to(torch.float32)  # For some reason hessian returns double
        acc = -(grad_val * (1 / (1 + grad_val.norm()**2)) * (v.T @ hess_val @ v)).flatten()
        return torch.cat([v, acc])

    def ode_fun_scipy(self, t, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
        new_state = self.ode_fun_torch(t, state_tensor)
        return new_state.to("cpu").detach().numpy()
   
    def expmap_scipy(self, theta, v):
        init = torch.cat([theta, v]).to("cpu").detach().numpy()
        solution = solve_ivp(self.ode_fun_scipy, [0, 1], init, dense_output=True, rtol=self.rtol, atol=self.atol)
        return solution
    
    def expmap_torchdiffeq(self, theta, v, num_ts=10):
        init = torch.cat([theta, v]).to("cpu").detach()
        ts = torch.linspace(0, 1, num_ts)
        solution = torchdiffeq.odeint(self.ode_fun_torch, init, ts , rtol=self.rtol, atol=self.atol)
        return solution

    def make_posterior_sample_scipy(self, n_samples=None):
        if n_samples is not None:
            self.make_posterior_sample_la(n_samples)
        n_samples = len(self.posterior_samples_la)
        self.posterior_samples = torch.zeros((n_samples, self.num_params))
        self.trajectories = []
        for i, la_sample in enumerate(self.posterior_samples_la):
            #print(f"{i = }, {la_sample = }")
            res = self.expmap_scipy(self.mean, la_sample)
            riemann_trajectory = torch.tensor(res.y[:self.num_params])
            self.trajectories.append(riemann_trajectory)
            riemann_sample = res.y[:self.num_params, -1]
            self.posterior_samples[i] = torch.tensor(riemann_sample)
        return self.posterior_samples

    def make_posterior_sample_torchdiffeq(self, n_samples=None):
        if n_samples is not None:
            self.make_posterior_sample_la(n_samples)
        n_samples = len(self.posterior_samples_la)
        self.posterior_samples = torch.zeros((n_samples, self.num_params))
        self.trajectories = []
        for i, la_sample in enumerate(self.posterior_samples_la):
            res = self.expmap_torchdiffeq(self.mean, la_sample)
            riemann_trajectory = res[:,:self.num_params].T
            self.trajectories.append(riemann_trajectory)
            riemann_sample = res[-1,:self.num_params]
            self.posterior_samples[i] = riemann_sample
        return self.posterior_samples
    
    def make_posterior_sample(self, n_samples=None):
        return self.make_posterior_sample_torchdiffeq(n_samples)


# Now, let us make a model from the banana function, which has the x/y coordinates as input
# and the banana function value as output
# With this, we will build a function that takes a parameter vector and an input tensor, and returns the model output tensor

banana_function = functional_banana(curvature=1.0, sigma_x=2.0, sigma_y=1.0)
banana_model = Model_from_func(banana_function, input_shape=[2])
torch.nn.utils.vector_to_parameters(torch.tensor([0.0, 0.0]), banana_model.parameters())
const_prior = lambda params: torch.tensor(0.0)
loss_fn = neglog_loss()

parametersubset = dict(banana_model.named_parameters())

xs = torch.tensor([0.0, 0.0]).unsqueeze(0)
ys = banana_function(xs[0]).unsqueeze(0)
params_init = torch.zeros(2)

R_sampler = Riemann_sampler(banana_model, parametersubset, xs=xs, ys=ys, loss_fn=loss_fn, prior_logprob=const_prior)
R_sampler.fit(fitting_type="hessian")

_=R_sampler.make_posterior_sample_la(10)

import timeit
runtime = timeit.timeit(lambda: R_sampler.make_posterior_sample_scipy(), number=10)
print(f"{runtime = }")
R_params = R_sampler.make_posterior_sample_scipy()
plot_traj(R_sampler)

runtime = timeit.timeit(lambda: R_sampler.make_posterior_sample_torchdiffeq(), number=10)
print(f"{runtime = }")
R_params_torchdiffeq = R_sampler.make_posterior_sample_torchdiffeq()
plot_traj(R_sampler)


# This also also works!
n_samples=3
R_sampler.LA_samples = R_sampler.make_posterior_sample_la(n_samples)
R_sampler.posterior_samples = torch.zeros((len(R_sampler.LA_samples), R_sampler.num_params))
R_sampler.trajectories = []
exp_map_vmap = vmap(R_sampler.expmap_stacked, in_dims=0)
stacked = torch.column_stack([R_sampler.mean.repeat(n_samples, 1), R_sampler.LA_samples])
res = exp_map_vmap(stacked)



logprob = lambda preds, target: torch.sum(preds.log())
sampler = MCMC_sampler(banana_model, parametersubset, xs=xs, ys=ys, loss_fn=logprob, prior_logprob=const_prior)
tens_params_hmc = sampler.make_posterior_sample(1000)
plt.scatter(tens_params_hmc[:, 0], tens_params_hmc[:, 1])
