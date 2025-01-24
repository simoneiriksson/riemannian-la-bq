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
import seaborn as sns
import pandas as pd



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
        #print(f"{theta = }")
        #print(f"{f_loss(theta) = }")
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
        return solution, ts

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
        self.trajectories_ts = []
        for i, la_sample in enumerate(self.posterior_samples_la):
            res, ts = self.expmap_torchdiffeq(self.mean, la_sample)
            riemann_trajectory = res[:,:self.num_params].T
            self.trajectories.append(riemann_trajectory)
            self.trajectories_ts.append(ts) 
            riemann_sample = res[-1,:self.num_params]
            self.posterior_samples[i] = riemann_sample
        return self.posterior_samples
    
    def make_posterior_sample(self, n_samples=None):
        return self.make_posterior_sample_torchdiffeq(n_samples)


def riemann_plotter(R_sampler, sample_markers=".", plot_traject=True, plot_traj_marker=".", max_samples=None, LA_arrows=[], kde=True):
    fig, ax = plt.subplots()
    cmap = plt.get_cmap("gist_rainbow")
    if plot_traject is True: plot_traject = range(len(R_sampler.trajectories))
    else: plot_traject=[]
    if LA_arrows is True: LA_arrows = range(len(R_sampler.trajectories))
    else: LA_arrows=[]
    if max_samples is None:
        max_samples = len(R_sampler.trajectories)
    if kde:
        df = pd.DataFrame(R_sampler.posterior_samples[:max_samples].numpy(), columns=["x", "y"])
        sns.kdeplot(data=df, x="x", y="y", fill=True, levels=10, color="black", ax=ax)
    if sample_markers is not None:
        ax.scatter(R_sampler.posterior_samples[:max_samples, 0], R_sampler.posterior_samples[:max_samples, 1], c="black")
    for i, sample in enumerate(R_sampler.trajectories[:max_samples]):
        color_no = (i*cmap.N)//len(R_sampler.trajectories[:max_samples])
        if (i in plot_traject) or (plot_traject is True):
            ax.plot(sample[0], sample[1], marker=plot_traj_marker, c=cmap(color_no))
        if (i in LA_arrows) or (LA_arrows is True):
            dx = R_sampler.posterior_samples_la[i][0] - R_sampler.mean[0]
            dy = R_sampler.posterior_samples_la[i][1] - R_sampler.mean[1]
            ax.arrow(R_sampler.mean[0].detach(), R_sampler.mean[1].detach(), dx.detach(), dy.detach(), head_width=0.1, head_length=0.1, color=cmap(color_no))
    return fig, ax
