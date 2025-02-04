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
from models import functional_banana, Model_from_func, functional_d1_halfcircle, functional_d1_2, functional_d1, functional_d1_fourth_degree_poly, LinearModel, functional_d1_normal
from matplotlib import pyplot as plt
from MCMC_sampler import MCMC_sampler
import torchdiffeq
import seaborn as sns
import pandas as pd
from riemann_sampler import Riemann_sampler, riemann_plotter

import GPy
from FullGaussianMeasure import FullGaussianMeasure
from emukit.model_wrappers.gpy_quadrature_wrappers import BaseGaussianProcessGPy, RBFGPy
from emukit.quadrature.kernels import QuadratureRBFLebesgueMeasure, QuadratureRBFGaussianMeasure
from emukit.quadrature.measures import LebesgueMeasure, GaussianMeasure

from emukit.quadrature.methods import VanillaBayesianQuadrature
from emukit.quadrature.acquisitions import IntegralVarianceReduction
from emukit.core.optimization import GradientAcquisitionOptimizer
from emukit.core.parameter_space import ParameterSpace
import numpy as np

from RayAcquisition import RayAcquisition

def reasonable_box_fixed(self_object, factor=4):
    def fn():
        lower = self_object.mean - factor * np.sqrt(self_object.variance)
        upper = self_object.mean + factor * np.sqrt(self_object.variance)
        return list(zip(lower, upper))
    return fn




class BayesianQuadrature_rays():
    def __init__(self, Rsampler: Riemann_sampler, evaluation_model, measure="gaussian_rescaled", 
                 integral_bounds_std=2, GP_lengthscale=1.0, GP_variance=1.0, num_timesteps=10, use_ray_acqusition=True, use_rays=True,
                 square_plots=True, theta_space_plot_limits=None, xs=None, parametersubset=None):
        self.Rsampler = Rsampler
        self.evaluation_model = evaluation_model
        self.measure = measure
        self.integral_bounds_std = integral_bounds_std
        self.GP_lengthscale = GP_lengthscale
        self.GP_variance = GP_variance
        self.theta_space_plot_limits = theta_space_plot_limits

        self.functional_fwd = make_functional_fwd_vector(evaluation_model, xs, parametersubset=parametersubset)

        self.num_timesteps = num_timesteps
        self.square_plots = square_plots

        if measure=="gaussian_rescaled":
            self.integrand = self.gaussian_integrand_rescaled
            self.measure_rescaled = True
            self.measure_type = "gaussian"

        elif measure == "lebesgue":
            self.integrand = self.lebesgue_integrand
            self.measure_rescaled = False
            self.measure_type = "lebesgue"

        elif measure == "lebesgue_rescaled":
            self.integrand = self.lebesgue_integrand_rescale
            self.measure_rescaled = True
            self.measure_type = "lebesgue"

        #self.v_init = torch.zeros(self.Rsampler.num_params)
        #self.model_output_init = self.y.unsqueeze(0).detach().numpy()
        self.sampling_dims = self.Rsampler.subspace_rank
        self.v_init = np.zeros((self.sampling_dims))
        self.y_init, self.theta_init = self.integrand(torch.tensor(self.v_init, dtype=torch.float32))
        self.y_init, self.theta_init = self.y_init[0,:].detach().numpy(), self.theta_init[0,:].detach().numpy()   
        
        self.vs = np.expand_dims(self.v_init, 0)
        self.vs_ray = self.vs
        self.integrand_values = np.expand_dims(self.y_init, 0)
        self.thetas = np.expand_dims(self.theta_init, 0)
        


    # GPy takes X and Y values at initialization. Those will be overwritten later when the emukit model is initialized.
        self.BQ_kernel = kernel=GPy.kern.RBF(
                                    input_dim=self.sampling_dims, 
                                    lengthscale=self.GP_lengthscale,
                                    variance=self.GP_variance)
        #print(f"{self.vs.shape = }, {self.model_output.shape = }")
        self.gpy_model = GPy.models.GPRegression(X=self.vs[0:1], 
                                                 Y=self.integrand_values[0:1], 
                                                 kernel=self.BQ_kernel)
        self.emukit_rbf = RBFGPy(self.gpy_model.kern)

        if self.measure_rescaled:
            self.limits = self.integral_bounds_std * np.ones(self.sampling_dims) # use dimensionality of the subspace model
            self.plot_limits = self.integral_bounds_std * np.ones(self.Rsampler.num_params) # use dimensionality of the full model

        elif not self.measure_rescaled:
            # set the limits for the integration bounds to the standard deviation of the laplace approximation times some number.
            if self.Rsampler.is_subspacelaplace:
                scaling = self.Rsampler.svd_S[:self.sampling_dims].sqrt().detach().numpy() # use dimensionality of the subspace model
                # perhaps this works?
            else:
                scaling  = np.sqrt(torch.diag(self.Rsampler.covariance).detach().numpy()) # use dimensionality of the full model
            self.limits = self.integral_bounds_std * scaling # use dimensionality of the sampling space (full or subspace)
            self.plot_limits = self.integral_bounds_std * np.sqrt(torch.diag(self.Rsampler.covariance).detach().numpy()[-self.sampling_dims:])  # use dimensionality of the subspace model
        if self.square_plots:
            self.plot_limits = self.plot_limits.max() * np.ones_like(self.plot_limits)

        if self.measure_type=="gaussian":
            self.emukit_measure = GaussianMeasure(mean=np.zeros(self.sampling_dims), 
                                              variance=np.ones(self.sampling_dims))
            self.emukit_measure.reasonable_box = reasonable_box_fixed(self.emukit_measure, factor=self.integral_bounds_std)
            self.emukit_qrbf = QuadratureRBFGaussianMeasure(self.emukit_rbf, self.emukit_measure)
            self.plt_measure_dist = torch.distributions.MultivariateNormal(torch.zeros(self.sampling_dims), torch.eye(self.sampling_dims))

        if self.measure_type=="lebesgue":
            print(f"{self.limits = }")
            print(f"{self.plot_limits = }")
            limits_list_of_lists = [[-l, l] for l in self.limits]
            self.emukit_measure = LebesgueMeasure.from_bounds(limits_list_of_lists)
            self.emukit_qrbf = QuadratureRBFLebesgueMeasure(self.emukit_rbf, self.emukit_measure)
            self.plt_measure_dist = torch.distributions.Uniform(-torch.tensor(self.plot_limits), torch.tensor(self.plot_limits))
            self.plt_measure_dist.log_prob = lambda z: -torch.log((torch.tensor(self.plot_limits) * 2).prod()).repeat(z.shape[0])

        self.emukit_model = BaseGaussianProcessGPy(kern=self.emukit_qrbf, gpy_model=self.gpy_model)
        self.emukit_method = VanillaBayesianQuadrature(base_gp=self.emukit_model, X=self.vs[0:1], Y=self.integrand_values[0:1])
        if use_ray_acqusition:
            self.ivr_acquisition = RayAcquisition(self.emukit_method, self.v_init, self.num_timesteps)
        else:
            self.ivr_acquisition = IntegralVarianceReduction(self.emukit_method)

        self.space = ParameterSpace(self.emukit_method.reasonable_box_bounds.convert_to_list_of_continuous_parameters())
        self.optimizer = GradientAcquisitionOptimizer(self.space)
        self.has_plotted = False
        self.use_rays = use_rays
        self.plot_N_mesh=100
        
    def step(self):
        v_new, acq_val = self.optimizer.optimize(self.ivr_acquisition)
        #print(f"{(v_new**2).sum() = }, {acq_val = }")
        v_new_t = torch.tensor(v_new).squeeze(0).to(torch.float32)
        #print(f"{v_new_t = }")
        self.vs = np.append(self.vs, v_new, axis=0)
        self.ys_new, thetas_new = self.integrand(v_new_t)
        #print(f"{self.ys_new = }, {thetas_new = }")

        if self.use_rays:
            use_index = range(1, self.num_timesteps)
        else:
            use_index = [self.num_timesteps-1]
        #print(f"{use_index = }")
        #print(f"{thetas_new.shape = }")
        #print(f"{thetas_new[use_index].shape = }")
        self.thetas = np.append(self.thetas, thetas_new[use_index].numpy(), axis=0)
    
        vs_ray_new = np.linspace(self.v_init, v_new[0], self.num_timesteps)

        self.vs_ray = np.append(self.vs_ray, vs_ray_new[use_index], axis=0)
        
        self.integrand_values = np.append(self.integrand_values, self.ys_new[use_index].detach().numpy(), axis=0)
        self.emukit_method.set_data(self.vs_ray, self.integrand_values)
        self.integral_mean, self.integral_variance = self.emukit_method.integrate()
        return self.integral_mean, self.integral_variance

    def lebesgue_integrand(self, v):
        num_ts = self.num_timesteps
        res, ts = self.Rsampler.expmap_torchdiffeq(self.Rsampler.mean, self.Rsampler.scale @ v, num_ts=num_ts)
        #theta = res[-1,:self.Rsampler.num_params].T
        thetas = res[:,:self.Rsampler.num_params]
        vs_ray = torch.tensor(np.linspace(self.v_init, v.detach().numpy(), self.num_timesteps))
        v_dist = torch.distributions.MultivariateNormal(torch.zeros(self.sampling_dims), torch.eye(self.sampling_dims))
        #ys = torch.stack([self.functional_fwd(theta) * self.Rsampler.posterior_logprob(vs_ray[i]).exp() for i, theta in enumerate(thetas)])
        #print(f"{vs_ray = }, {thetas = }")
        ys = torch.stack([self.functional_fwd(theta) * v_dist.log_prob(vs_ray[i]).exp() for i, theta in enumerate(thetas)])
        #return self.functional_fwd(theta) * R_sampler.posterior_logprob(v).exp(), theta
        return ys, thetas

    def lebesgue_integrand_rescale(self, z):
        num_ts = self.num_timesteps
        g = lambda z: self.Rsampler.scale @ z
        v = g(z)
        res, ts = self.Rsampler.expmap_torchdiffeq(self.Rsampler.mean, v, num_ts=num_ts)
        #vs_ray = torch.tensor(np.linspace(self.v_init, v.detach().numpy(), self.num_timesteps))
        z_init = np.zeros(self.sampling_dims)
        zs_ray = torch.tensor(np.linspace(z_init, z.detach().numpy(), self.num_timesteps))
        #print(f"{vs_ray.shape = }, {zs_ray.shape = }")
        #zs_ray = torch.stack([g_inv(v) for v in vs_ray])
        #theta = res[-1,:self.Rsampler.num_params].T
        thetas = res[:,:self.Rsampler.num_params]
        z_dist = torch.distributions.MultivariateNormal(torch.zeros(self.sampling_dims), torch.eye(self.sampling_dims))
        ys = torch.stack([self.functional_fwd(theta) * z_dist.log_prob(zs_ray[i]).exp() for i, theta in enumerate(thetas)])
        #print(f"{ys.shape = }, {thetas.shape = }")
        return ys, thetas

    def gaussian_integrand_rescaled(self, z):
        num_ts = self.num_timesteps
        g = lambda z: self.Rsampler.scale @ z
        v = g(z)
        res, ts = self.Rsampler.expmap_torchdiffeq(self.Rsampler.mean, v, num_ts=num_ts)
        thetas = res[:,:self.Rsampler.num_params]
        ys = torch.stack([self.functional_fwd(theta) for theta in thetas])
        return ys, thetas

    def plot_integrand_2d(self, ax=None):
        # plot integrand values
        ax.scatter(self.vs_ray[:,0], self.vs_ray[:,1], c=self.integrand_values, cmap="viridis")
        ax.set_title("Integrant values")
        ax.set_xlim(-self.plot_limits[0], self.plot_limits[0])
        ax.set_ylim(-self.plot_limits[1], self.plot_limits[1])
        
    def plot_integrand_measure_2d(self, ax=None):
        ax.scatter(self.vs_ray[:,0], self.vs_ray[:,1], 
                    c=self.integrand_values[:,0] * self.plt_measure_dist.log_prob(torch.tensor(self.vs_ray)).exp().detach().numpy(), 
                    cmap="viridis")
        ax.set_title("Integrant values times measure")
        ax.set_xlim(-self.plot_limits[0], self.plot_limits[0])
        ax.set_ylim(-self.plot_limits[1], self.plot_limits[1])

    def plt_BQ_mean_2d(self, ax=None):
        self.plot_init_2d()
        mu_plot, var_plot = self.emukit_method.predict(self.xys_plt)
        ax.contourf(self.xs_plt, self.ys_plt, mu_plot.reshape(self.plot_N_mesh, self.plot_N_mesh).T, cmap="viridis")
        ax.scatter(self.vs_ray[:,0], self.vs_ray[:,1], c="k", marker="x")
        ax.set_title("BQ Model mean")
        ax.set_xlim(-self.plot_limits[0], self.plot_limits[0])
        ax.set_ylim(-self.plot_limits[1], self.plot_limits[1])
    
    def plot_BQ_mean_measure_2d(self, ax=None):
        self.plot_init_2d()
        mu_plot, var_plot = self.emukit_method.predict(self.xys_plt)
        ax.contourf(self.xs_plt, self.ys_plt, mu_plot.reshape(self.plot_N_mesh,self.plot_N_mesh).T * self.plt_measure_dist.log_prob(torch.tensor(self.xys_plt.reshape(self.plot_N_mesh, self.plot_N_mesh, -1))).exp().detach().numpy(), cmap="viridis")
        ax.scatter(self.vs_ray[:,0], self.vs_ray[:,1], c="k", marker="x")
        ax.set_title("BQ Model mean times measure")

    def plot_BQ_std_2d(self, ax=None):
        self.plot_init_2d()
        mu_plot, var_plot = self.emukit_method.predict(self.xys_plt)
        ax.contourf(self.xs_plt, self.ys_plt, np.sqrt(var_plot).reshape(self.plot_N_mesh,self.plot_N_mesh).T, cmap="viridis")
        ax.scatter(self.vs_ray[:,0], self.vs_ray[:,1], c="k", marker="x")
        #axes[3].scatter(xys_plt[:,0], xys_plt[:,1], c=np.sqrt(var_plot), cmap="viridis")
        ax.set_title("BQ Model std")

    def plot_acquisition_2d(self, ax=None):
        self.plot_init_2d()
        squared_correlation, integral_current_var, y_predictive_var, predictive_cov = self.ivr_acquisition._evaluate(self.xys_plt)
        ax.contourf(self.xs_plt, self.ys_plt, np.sqrt(squared_correlation).reshape(self.plot_N_mesh,self.plot_N_mesh).T, cmap="viridis")
        ax.scatter(self.vs_ray[:,0], self.vs_ray[:,1], c="k", marker="x")
        ax.set_title("Acquisition function")

    def plot_true_function_likelihood_2d(self, ax=None):
        self.plot_init_2d()
        ax.contourf(self.xs_plt, self.ys_plt, self.function_times_likelihood.detach(), cmap="viridis")
        ax.scatter(self.thetas[:,0] - self.Rsampler.mean[0].detach().numpy(), self.thetas[:,1] - self.Rsampler.mean[1].detach().numpy(), c="k", marker="x")
        ax.set_title("True function times model likelihood")
        ax.set_xlim(-self.plot_limits[0], self.plot_limits[0])
        ax.set_ylim(-self.plot_limits[1], self.plot_limits[1])

    def plot_model_likelihood_2d(self, ax=None):
        self.plot_init_2d()
        ax.contourf(self.xs_plt, self.ys_plt, (-self.plt_model_loss*2).exp().detach(), cmap="viridis")
        ax.contour(self.xs_plt, self.ys_plt, self.function_vals.detach(), cmap="Grays")
        
        ax.scatter(self.thetas[:,0] - self.Rsampler.mean[0].detach().numpy(), self.thetas[:,1] - self.Rsampler.mean[1].detach().numpy(), c="k", marker="x")
        #axes[0][3].contour(self.xs_plt, self.ys_plt, self.function_vals.detach(), c="k")
        ax.set_title("Model loss and true function values")
        ax.set_xlim(-self.plot_limits[0], self.plot_limits[0])
        ax.set_ylim(-self.plot_limits[1], self.plot_limits[1])


    def plot_init_2d(self):
        if not self.has_plotted:
            self.xs_plt = np.linspace(-self.plot_limits[0], self.plot_limits[0], self.plot_N_mesh)
            self.ys_plt = np.linspace(-self.plot_limits[1], self.plot_limits[1], self.plot_N_mesh)
            self.xys_plt = np.array([[x, y] for x in self.xs_plt for y in self.ys_plt])

            self.function_vals = torch.stack([self.functional_fwd(obs + self.Rsampler.mean) for obs in torch.tensor(self.xys_plt, dtype=torch.float32)]).reshape(self.plot_N_mesh, self.plot_N_mesh).T
            
            #f_model = make_functional_fwd_vector(self.Rsampler.model, self.xs, parametersubset=dict(self.Rsampler.model.named_parameters()))
            #f_model = make_functional_fwd_xs(self.Rsampler.model)
            #self.f_loss = functional_loss_for_vmap(f_model, self.Rsampler.parametersubset, self.Rsampler.loss_fn, self.Rsampler.xs, self.Rsampler.ys, prior_logprob=self.Rsampler.prior_logprob)
            
            self.plt_model_loss = torch.stack([self.Rsampler.f_loss(obs + self.Rsampler.mean) for obs in torch.tensor(self.xys_plt, dtype=torch.float32)]).reshape(self.plot_N_mesh, self.plot_N_mesh).T
            self.function_times_likelihood = self.function_vals * (-self.plt_model_loss).exp()


    def plot_integrand_measure_1d(self, ax=None):
        xs_plt = np.linspace(-self.limits, self.limits, 100)
        mu_plot, var_plot = self.emukit_method.predict(xs_plt)

        # plot BQ model mean times measure
        plt_measure = self.plt_measure_dist.log_prob(torch.tensor(xs_plt)).exp().detach().numpy()
        plt_measure_points = self.plt_measure_dist.log_prob(torch.tensor(self.vs_ray)).exp().detach().numpy()
        ax.fill_between(xs_plt[:,0], (mu_plot - np.sqrt(var_plot))[:,0]* plt_measure, (mu_plot + np.sqrt(var_plot))[:,0]* plt_measure, alpha=0.5, label="BQ std")
        ax.scatter(self.vs_ray[:,0], self.integrand_values[:,0] * plt_measure_points , c="k", marker="x", label="Integrant times measure")
        ax.plot(xs_plt[:,0], mu_plot[:,0] * plt_measure, label="BQ mean times measure", c="b")
        ax.plot(xs_plt[:,0], plt_measure, label="Measure", c="g")
        ax.set_title("Integrand times measure")
        ax.legend()

    def plot_integrand_1d(self, ax=None):
        self.plot_init_1d()
        # plot integrand values
        xs_plt = np.linspace(-self.limits, self.limits, self.plot_N_mesh)
        mu_plot, var_plot = self.emukit_method.predict(xs_plt)
        ax.fill_between(xs_plt[:,0], (mu_plot - np.sqrt(var_plot))[:,0], (mu_plot + np.sqrt(var_plot))[:,0], alpha=0.5, label="BQ std")
        ax.scatter(self.vs_ray[:,0], self.integrand_values, c="k", marker="x", label="Integrant")
        ax.plot(xs_plt, mu_plot, label="BQ mean", c="b")
        ax.set_title("Integrand values")
        #ax.set_xlim(-self.limits[0], self.limits[0])
        #ax.set_ylim(self.plt_1d_ylimit[0], self.plt_1d_ylimit[1])
        ax.legend()

    def plot_true_function_likelihood_1d(self, ax=None):
        self.plot_init_1d()
        ax.plot(self.xs_plt_1d[:,0], self.function_times_likelihood.detach().numpy(), label="True function times model likelihood")
        ax.plot(self.xs_plt_1d[:,0], self.function_vals.detach().numpy(), label="True function")
        ax.plot(self.xs_plt_1d[:,0], self.plt_likelihood.detach().numpy(), label="Model likelihood")

        ax.scatter(self.thetas[:,0] - self.Rsampler.mean[0].detach().numpy(), self.integrand_values, c="k", marker="x", label="Integrant")
        ax.set_title("True function times model likelihood")
        ax.set_xlim(-self.limits, self.limits)
        ax.set_ylim(min(self.function_times_likelihood.nan_to_num(posinf=10).min().detach(), self.integrand_values.min()), 
                    max(self.function_times_likelihood.nan_to_num(posinf=10).max().detach(), self.integrand_values.max()) 
                    )
        ax.set_ylim(0,.5)
        if self.theta_space_plot_limits is not None:
            ax.set_xlim(self.theta_space_plot_limits)
        ax.legend()

    def plot_init_1d(self):
        self.plt_1d_ylimit = [self.integrand_values.min(), self.integrand_values.max()]
        self.xs_plt_1d = np.linspace(-self.limits, self.limits, self.plot_N_mesh)
        self.function_vals = torch.stack([self.functional_fwd(obs + self.Rsampler.mean) for obs in torch.tensor(self.xs_plt_1d)]).reshape(self.plot_N_mesh)
        self.plt_model_loss = torch.stack([self.Rsampler.f_loss(obs + self.Rsampler.mean) for obs in torch.tensor(self.xs_plt_1d)])
        self.plt_likelihood = (-self.plt_model_loss).exp()
        self.function_times_likelihood = self.function_vals * self.plt_likelihood
       
    def plot(self):
        if self.Rsampler.num_params == 2 and self.sampling_dims == 2: # This is the case where we have 2d parameters and 2d subspace
            fig, axes = plt.subplots(2,4, figsize=(20,10))
            self.plot_integrand_2d(ax=axes[0][0])
            self.plot_integrand_measure_2d(ax=axes[1][0])
            self.plt_BQ_mean_2d(ax=axes[0][1])
            self.plot_BQ_mean_measure_2d(ax=axes[1][1])
            self.plot_model_likelihood_2d(ax=axes[0][2])
            self.plot_true_function_likelihood_2d(ax=axes[1][2])
            self.plot_BQ_std_2d(ax=axes[0][3])
            self.plot_acquisition_2d(ax=axes[1][3])
            self.has_plotted = True
            return fig, axes

        if self.Rsampler.num_params not in  [1,2] and self.sampling_dims == 2: # This is the case where we have >2d parameters and 2d subspace
            fig, axes = plt.subplots(2,4, figsize=(20,10))
            self.plot_integrand_2d(ax=axes[0][0])
            self.plot_integrand_measure_2d(ax=axes[0][1])
            self.plt_BQ_mean_2d(ax=axes[1][0])
            self.plot_BQ_mean_measure_2d(ax=axes[1][1])
            self.plot_BQ_std_2d(ax=axes[1][2])
            self.plot_acquisition_2d(ax=axes[1][3])
            self.has_plotted = True
            return fig, axes


        elif self.Rsampler.num_params == 2 and self.sampling_dims == 1: # This is the case where we have 2d parameters and 1d subspace
            fig, axes = plt.subplots(1,3, figsize=(15,5))
            self.plot_integrand_1d(ax=axes[0])
            self.plot_integrand_measure_1d(ax=axes[1])
            self.plot_true_function_likelihood_2d(ax=axes[2])
            self.plot_model_likelihood_2d(ax=axes[3])
            return fig, axes

        elif self.Rsampler.num_params == 1 and self.sampling_dims == 1: # This is the case where we have 1d parameters and 1d subspace
            fig, axes = plt.subplots(1,3, figsize=(15,5))
            self.plot_integrand_1d(ax=axes[0])
            self.plot_integrand_measure_1d(ax=axes[1])
            self.plot_true_function_likelihood_1d(ax=axes[2])
            return fig, axes

