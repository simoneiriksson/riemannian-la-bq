import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.distributions.multivariate_normal import _precision_to_scale_tril
from utils import tensify, loss_func_from_target_sigma, make_functional_fwd_xs, vector_to_parameterdict, make_functional_fwd
from GGN_hessian import GGN_hessian_from_loader
from hessian import hessian_from_model_loss_and_data, hessian_dict_to_matrix, hessian_from_loader, hessian_from_func
from riemannian_la.utils import NegLogLik_regression, NegLogLik_classification, iid_gaussian_prior
from torch.func import grad, jvp, vjp, hessian, jacfwd, jacrev, vmap, functional_call
from laplace_approx import Laplace
from scipy.integrate import solve_ivp
from utils import make_functional_fwd_xs, functional_loss_for_vmap, neglog_loss, make_functional_fwd_vector
from models import functional_banana, Model_from_func, functional_d1_halfcircle
from matplotlib import pyplot as plt
from MCMC_sampler import MCMC_sampler
import torchdiffeq
import seaborn as sns
import pandas as pd
from riemann_sampler import Riemann_sampler, riemann_plotter

from emukit.model_wrappers.gpy_quadrature_wrappers import BaseGaussianProcessGPy, RBFGPy
from emukit.quadrature.kernels import QuadratureRBFLebesgueMeasure, QuadratureRBFGaussianMeasure
from emukit.quadrature.measures import LebesgueMeasure
from FullGaussianMeasure import FullGaussianMeasure
import GPy

from emukit.quadrature.methods import VanillaBayesianQuadrature
from emukit.quadrature.acquisitions import IntegralVarianceReduction
from emukit.core.optimization import GradientAcquisitionOptimizer
from emukit.core.parameter_space import ParameterSpace
import numpy as np

n_mesh = 1000
a = 1.0

#################
# Now for 1d test
func_1d = functional_d1_halfcircle(a=a)
func_1d_model = Model_from_func(func_1d, input_shape=[1])
loss_fn = lambda preds, target: torch.sum()
xs = torch.tensor([0.0]).unsqueeze(0)
ys = func_1d(xs[0]).unsqueeze(0)

const_prior = lambda params: torch.tensor(1.0)

class tiny_ridiculess_model_class(torch.nn.Module):
    def __init__(self, n_params):
        super().__init__()
        self.params = torch.nn.Parameter(torch.arange(n_params).float())
    def forward(self, x):
        #return torch.sin(self.params*3.1).sum().repeat(x.shape[0], 1)**2
        #return torch.ones(x.shape[0], 1)
        return self.params.sum().repeat(x.shape[0], 1)**2
        
evaluation_model = tiny_ridiculess_model_class(n_params=2)
functional_evaluation_model = make_functional_fwd_xs(evaluation_model)  # make functional version of model

parametersubset = dict(func_1d_model.named_parameters())

xs = torch.tensor([0.0]).unsqueeze(0)
xs.shape
X_init = xs.numpy()
ys = func_1d_model(xs[0]).unsqueeze(0)

torch.nn.utils.vector_to_parameters(torch.tensor([0.0]), func_1d_model.parameters())

functional_fwd = make_functional_fwd_vector(evaluation_model, xs, parametersubset=parametersubset)
R_sampler = Riemann_sampler(func_1d_model, parametersubset, xs=xs, ys=ys, loss_fn=neglog_loss(), prior_logprob=const_prior)
R_sampler.fit(fitting_type="hessian")


#lower_bound = (R_sampler.mean - R_sampler.covariance.diag()*10)
#upper_bound = (R_sampler.mean + R_sampler.covariance.diag()*10)
plot = True

#####################

def integrand(v):
    num_ts=10
    res, ts = R_sampler.expmap_torchdiffeq(R_sampler.mean, v, num_ts=num_ts)
    theta = res[-1,:R_sampler.num_params].T
    return functional_fwd(theta) * R_sampler.posterior_logprob(v).exp(), theta

y, theta_init = integrand(torch.tensor(X_init[0]))
Y_init = y.unsqueeze(0).detach().numpy()

# GPy takes X and Y values at initialization. Those will be overwritten later when the emukit model is initialized.
gpy_model = GPy.models.GPRegression(X=X_init, Y=Y_init, kernel=GPy.kern.RBF(
                        input_dim=X_init.shape[1], lengthscale=0.5, variance=1.0))

lower_bound = [-a*3]
upper_bound = [a*3]
integral_bounds = [(lower_bound[i], upper_bound[i]) for i in range(len(xs[0]))]
emukit_rbf = RBFGPy(gpy_model.kern)
emukit_measure = LebesgueMeasure.from_bounds(integral_bounds)
emukit_qrbf = QuadratureRBFLebesgueMeasure(emukit_rbf, emukit_measure)
emukit_model = BaseGaussianProcessGPy(kern=emukit_qrbf, gpy_model=gpy_model)

emukit_measure.reasonable_box()
dir(emukit_measure)


emukit_method = VanillaBayesianQuadrature(base_gp=emukit_model, X=X_init, Y=Y_init)
integral_mean, integral_variance = emukit_method.integrate()
print(f"Integral mean: {integral_mean}, integral std: {np.sqrt(integral_variance)}")

ivr_acquisition = IntegralVarianceReduction(emukit_method)

space = ParameterSpace(emukit_method.reasonable_box_bounds.convert_to_list_of_continuous_parameters())
optimizer = GradientAcquisitionOptimizer(space)


x_plot = np.linspace(lower_bound[0], upper_bound[0], 1000).reshape(-1, 1)
x_plot_int = torch.linspace(lower_bound[0], upper_bound[0], 20).reshape(-1, 1)
meh = np.array([meh.item() for x in x_plot_int for meh in integrand(x)]).reshape(-1, 2)
y_plot = meh[:, 0]
thetas_plot = meh[:, 1]

Xs = X_init
Ys = Y_init
thetas = theta_init.detach().numpy()
for i in range(20):
    x_new,_ = optimizer.optimize(ivr_acquisition)
    x_new_t = torch.tensor(x_new).squeeze(0).to(torch.float32)
    #res, ts = R_sampler.expmap_torchdiffeq(R_sampler.mean, x_new_t, num_ts=10)
    #theta = res[-1,:R_sampler.num_params].T
    y_new, theta = integrand(x_new_t)
    thetas = np.append(thetas, theta.item())
    Xs = np.append(Xs, x_new, axis=0)
    Ys = np.append(Ys, y_new.unsqueeze(0).detach().numpy(), axis=0)
    emukit_method.set_data(Xs, Ys)
    integral_mean, integral_variance = emukit_method.integrate()
    print(f"{i = }, Integral mean: {integral_mean}, integral std: {np.sqrt(integral_variance)}")

    if plot:
        mu_plot, var_plot = emukit_method.predict(x_plot)
        LEGEND_SIZE = 15/2
        FIGURE_SIZE = (12/2, 8/2)
        fig, axes = plt.subplots(1,1, figsize=(5,5))
        #plt.figure(figsize=FIGURE_SIZE)
        ax1 = axes
        ax1.plot(Xs, Ys, "ro", markersize=10, label="Observations")
        ax1.plot(x_plot_int, y_plot, "k", label="The Integrand")
        ax1.plot(x_plot, mu_plot, "C0", label="Model")
        ax1.fill_between(x_plot[:, 0],
                        mu_plot[:, 0] + np.sqrt(var_plot)[:, 0],
                        mu_plot[:, 0] - np.sqrt(var_plot)[:, 0], color="C0", alpha=0.6)
        ax1.fill_between(x_plot[:, 0],
                        mu_plot[:, 0] + 2 * np.sqrt(var_plot)[:, 0],
                        mu_plot[:, 0] - 2 * np.sqrt(var_plot)[:, 0], color="C0", alpha=0.4)
        ax1.fill_between(x_plot[:, 0],
                        mu_plot[:, 0] + 3 * np.sqrt(var_plot)[:, 0],
                        mu_plot[:, 0] - 3 * np.sqrt(var_plot)[:, 0], color="C0", alpha=0.2)
        ax1.legend(loc=2, prop={'size': LEGEND_SIZE})
        ax1.set_xlabel(r"$x$")
        #ax.xlabel(r"$x$")
        ax1.set_ylabel(r"$f(x)$")
        ax1.grid(True)
        ax1.set_xlim(lower_bound[0], upper_bound[0])
        ax1.set_ylim(-1, 1)
        plt.show()

plt.plot(Xs[:,0], thetas, "ro", markersize=10, label="Observations")

# num_ts=10
# for v_ in Xs:
#     v = torch.tensor(v_).to(torch.float32)
#     res, ts = R_sampler.expmap_torchdiffeq(R_sampler.mean, v, num_ts=num_ts)
#     theta = res[-1,:R_sampler.num_params].T
#     print(f"\n{v = }")
#     print(f"{theta = }")
#     print(f"{functional_fwd(theta) = }")
#     print(f"{R_sampler.posterior_logprob(v).exp() = }")
#     print(f"{integrand(v) = }")







#####################################
# Above, we integrated over a lebesque measure.
# Now, we try to use the Laplace posterior approximation as the measure.



def integrand(v):
    num_ts=10
    res, ts = R_sampler.expmap_torchdiffeq(R_sampler.mean, v, num_ts=num_ts)
    theta = res[-1,:R_sampler.num_params].T
    return functional_fwd(theta), theta

y, theta_init = integrand(torch.tensor(X_init[0]))
Y_init = y.unsqueeze(0).detach().numpy()
lower_bound = [-a*3]
upper_bound = [a*3]

# GPy takes X and Y values at initialization. Those will be overwritten later when the emukit model is initialized.
gpy_model = GPy.models.GPRegression(X=X_init, Y=Y_init, kernel=GPy.kern.RBF(
                        input_dim=X_init.shape[1], lengthscale=0.5, variance=1.0))

emukit_rbf = RBFGPy(gpy_model.kern)
emukit_measure = FullGaussianMeasure(mean=R_sampler.mean.detach().numpy(), variance=R_sampler.covariance.detach().numpy())
emukit_qrbf = QuadratureRBFGaussianMeasure(emukit_rbf, emukit_measure)
emukit_model = BaseGaussianProcessGPy(kern=emukit_qrbf, gpy_model=gpy_model)
emukit_method = VanillaBayesianQuadrature(base_gp=emukit_model, X=X_init, Y=Y_init)

emukit_measure.reasonable_box()
emukit_measure.variance

integral_mean, integral_variance = emukit_method.integrate()
print(f"Integral mean: {integral_mean}, integral std: {np.sqrt(integral_variance)}")


ivr_acquisition = IntegralVarianceReduction(emukit_method)
space = ParameterSpace(emukit_method.reasonable_box_bounds.convert_to_list_of_continuous_parameters())
optimizer = GradientAcquisitionOptimizer(space)

x_plot = np.linspace(lower_bound[0], upper_bound[0], 1000).reshape(-1, 1)
x_plot_int = torch.linspace(lower_bound[0], upper_bound[0], 20).reshape(-1, 1)
meh = np.array([meh.item() for x in x_plot_int for meh in integrand(x)]).reshape(-1, 2)
y_plot = meh[:, 0]
thetas_plot = meh[:, 1]



Xs = X_init
Ys = Y_init
thetas = theta_init.detach().numpy()

for i in range(10):
    x_new,_ = optimizer.optimize(ivr_acquisition)
    x_new_t = torch.tensor(x_new).squeeze(0).to(torch.float32)
    #res, ts = R_sampler.expmap_torchdiffeq(R_sampler.mean, x_new_t, num_ts=10)
    #theta = res[-1,:R_sampler.num_params].T
    y_new, theta = integrand(x_new_t)
    thetas = np.append(thetas, theta.item())
    Xs = np.append(Xs, x_new, axis=0)
    Ys = np.append(Ys, y_new.unsqueeze(0).detach().numpy(), axis=0)
    emukit_method.set_data(Xs, Ys)
    integral_mean, integral_variance = emukit_method.integrate()
    print(f"{i = }, Integral mean: {integral_mean}, integral std: {np.sqrt(integral_variance)}")

    if plot:
        mu_plot, var_plot = emukit_method.predict(x_plot)
        measure_plot = emukit_method.measure.compute_density(x_plot)
        LEGEND_SIZE = 15/2
        FIGURE_SIZE = (12/2, 8/2)
        fig, axes = plt.subplots(1,1, figsize=(5,5))
        #plt.figure(figsize=FIGURE_SIZE)
        ax1 = axes
        ax1.plot(Xs, Ys, "ro", markersize=10, label="Observations")
        ax1.plot(x_plot_int, y_plot, "k", label="The Integrand")
        ax1.plot(x_plot, measure_plot, "g", label="The Measure")
        ax1.plot(x_plot, measure_plot*mu_plot[:,0], "y", label="The Measure * Model")
        ax1.plot(x_plot, mu_plot, "C0", label="Model")
        ax1.fill_between(x_plot[:, 0],
                        mu_plot[:, 0] + np.sqrt(var_plot)[:, 0],
                        mu_plot[:, 0] - np.sqrt(var_plot)[:, 0], color="C0", alpha=0.6)
        ax1.fill_between(x_plot[:, 0],
                        mu_plot[:, 0] + 2 * np.sqrt(var_plot)[:, 0],
                        mu_plot[:, 0] - 2 * np.sqrt(var_plot)[:, 0], color="C0", alpha=0.4)
        ax1.fill_between(x_plot[:, 0],
                        mu_plot[:, 0] + 3 * np.sqrt(var_plot)[:, 0],
                        mu_plot[:, 0] - 3 * np.sqrt(var_plot)[:, 0], color="C0", alpha=0.2)
        ax1.legend(loc=2, prop={'size': LEGEND_SIZE})
        ax1.set_xlabel(r"$x$")
        #ax.xlabel(r"$x$")
        ax1.set_ylabel(r"$f(x)$")
        ax1.grid(True)
        ax1.set_xlim(lower_bound[0], upper_bound[0])
        max_y = max(measure_plot.max(), mu_plot.max())
        min_y = min(measure_plot.min(), mu_plot.min())
        ax1.set_ylim(min_y*1.5, max_y*1.5)
        plt.show()