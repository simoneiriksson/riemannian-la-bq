import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.distributions.multivariate_normal import _precision_to_scale_tril
from utils import loss_func_from_target_sigma, make_functional_fwd_xs, vector_to_parameterdict, make_functional_fwd, make_functional_fwd_vector_xs
from GGN_hessian import GGN_hessian_from_loader
from hessian import hessian_from_model_loss_and_data, hessian_dict_to_matrix, hessian_from_loader, hessian_from_func
from utils import NegLogLik_regression, NegLogLik_classification, iid_gaussian_prior_loss
from torch.func import grad, jvp, vjp, hessian, jacfwd, jacrev, vmap, functional_call
from laplace_approx import Laplace
from scipy.integrate import solve_ivp
from utils import make_functional_fwd_xs, functional_loss_for_vmap, neglog_loss, make_functional_fwd_vector, identity_func
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
from emukit.model_wrappers import GPyModelWrapper

from emukit.quadrature.kernels import QuadratureRBFLebesgueMeasure, QuadratureRBFGaussianMeasure, QuadratureRBF
from emukit.quadrature.measures import LebesgueMeasure, GaussianMeasure

from emukit.quadrature.methods import VanillaBayesianQuadrature, WarpedBayesianQuadratureModel, BoundedBayesianQuadrature
from emukit.quadrature.methods.warpings import SquareRootWarping, IdentityWarping
from emukit.quadrature.acquisitions import IntegralVarianceReduction
from emukit.core.optimization import GradientAcquisitionOptimizer
from emukit.core.parameter_space import ParameterSpace
from emukit.quadrature.interfaces.base_gp import IBaseGaussianProcess
from emukit.quadrature.kernels import GaussianEmbedding, LebesgueEmbedding, QuadratureKernel
from emukit.quadrature.interfaces.standard_kernels import IRBF

import numpy as np

from RayAcquisition import RayAcquisition

from typing import Tuple

import numpy as np



def reasonable_box_fixed(self_object, factor=4):
    def fn():
        lower = self_object.mean - factor * np.sqrt(self_object.variance)
        upper = self_object.mean + factor * np.sqrt(self_object.variance)
        return list(zip(lower, upper))
    return fn



def transform(type="sqrt", param=1):
    if type == "sqrt":
        def fn_forward(x):
            return (x).sqrt()
        def fn_backward(mus, vars):
            return [mus**2 + vars]
        
    elif type == "log":
        def fn_forward(x):
            return (x).log()
        def fn_backward(mus, vars):
            return [torch.exp(mus + .5*vars)]
        
    elif type == "softplus":
        def fn_forward(x):
            return (x+param).log()
        def fn_backward(mus, vars):
            return [(mus + 0.5*vars).exp()-param]
        
    elif type == "identity":
        def fn_forward(x):
            return x
        def fn_backward(mus, vars):
            return [mus, vars]
        
    elif type == "exp":
        def fn_forward(x):
            return x.exp()
        def fn_backward(mus, vars):
            return [mus.log()]

    elif type == "inv":
        def fn_forward(x):
            return 1/x
        def fn_backward(mus, vars):
            return [1/mus]
        
    elif type == "softexp":
        def fn_forward(x):
            return (x).exp()-param
        def fn_backward(mus, vars):
            return [(mus + vars*.5).exp()-param]
        
    elif type == "power":
        def fn_forward(x):
            return (x)**(param)
        def fn_backward(mus, vars):
            return [(mus**2)**(1/(param*2)) + .5*vars]

    return fn_forward, fn_backward

class VanillaBayesianQuadrature_multidim_output(WarpedBayesianQuadratureModel):
    def __init__(self, base_gp: IBaseGaussianProcess, X: np.ndarray, Y: np.ndarray):
        super(VanillaBayesianQuadrature_multidim_output, self).__init__(base_gp=base_gp, warping=IdentityWarping(), X=X, Y=Y)

    def predict_base(self, X_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        m, cov = self.base_gp.predict(X_pred)
        return m, cov, m, cov

    def predict_base_with_full_covariance(self, X_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        m, cov = self.base_gp.predict_with_full_covariance(X_pred)
        return m, cov, m, cov

    def integrate(self) -> Tuple[float, float]:
        kernel_mean_X = self.base_gp.kern.qK(self.X)
        integral_mean = np.dot(kernel_mean_X, self.base_gp.graminv_residual())[0, :]
        integral_var = self.base_gp.kern.qKq() - (kernel_mean_X @ self.base_gp.solve_linear(kernel_mean_X.T))[0, 0]
        return integral_mean, integral_var

    def get_prediction_gradients(self, X: np.ndarray) -> Tuple:
        return self.base_gp.get_prediction_gradients(X)


class BayesianQuadrature_rays():
    def __init__(self, Rsampler: Riemann_sampler, evaluation_model=None, measure="gaussian_rescaled", 
                 integral_bounds_std=2, GP_lengthscale=1.0, GP_variance=1.0, num_timesteps=10, 
                 use_ray_acqusition=True, use_rays=True,
                 square_plots=True, theta_space_plot_limits=None, xs=None, parametersubset=None, output_func=identity_func, device="cpu", contour_fill="Grey", contour_linewidth=.1):
        self.contour_linewidth = contour_linewidth
        self.contour_fill = contour_fill
        self.device=device
        self.Rsampler = Rsampler
        self.evaluation_model = evaluation_model
        self.measure = measure
        self.integral_bounds_std = integral_bounds_std
        self.GP_lengthscale = GP_lengthscale
        self.GP_variance = GP_variance
        self.theta_space_plot_limits = theta_space_plot_limits
        if parametersubset is None:
            self.parametersubset = dict(evaluation_model.named_parameters())
        else:
            self.parametersubset = parametersubset


        self.output_func = output_func
        self.functional_fwd = make_functional_fwd_vector(evaluation_model, xs, parametersubset=self.parametersubset, output_func=output_func)
        self.functional_fwd_xs = make_functional_fwd_xs(evaluation_model, output_func=output_func)

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
        self.v_init = torch.zeros((self.sampling_dims), dtype=torch.float32).to(self.device)
        self.y_init, self.theta_init = self.integrand(self.v_init)
        self.y_init, self.theta_init = self.y_init[0,:], self.theta_init[0,:]
        
        #self.vs = np.expand_dims(self.v_init, 0)
        self.vs = self.v_init.unsqueeze(0)
        self.vs_ray = self.vs
        #self.integrand_values = np.expand_dims(self.y_init, 0)
        self.integrand_values = self.y_init.unsqueeze(0)
        #self.thetas = np.expand_dims(self.theta_init, 0)
        self.thetas = self.theta_init.unsqueeze(0)
        self.steps = 0


    # GPy takes X and Y values at initialization. Those will be overwritten later when the emukit model is initialized.
        self.BQ_kernel = kernel=GPy.kern.RBF(
                                    input_dim=self.sampling_dims, 
                                    lengthscale=self.GP_lengthscale,
                                    variance=self.GP_variance)
        #print(f"{self.vs.shape = }, {self.integrand_values.shape = }")
        self.gpy_model = GPy.models.GPRegression(X=self.vs_ray.detach().cpu().numpy(), 
                                                 Y=self.integrand_values.detach().cpu().numpy(), 
                                                 kernel=self.BQ_kernel)
        self.emukit_rbf = RBFGPy(self.gpy_model.kern)

        if self.measure_rescaled:
            self.limits = self.integral_bounds_std * torch.ones(self.sampling_dims).to(self.device) # use dimensionality of the subspace model
            self.plot_limits = self.integral_bounds_std * torch.ones(self.Rsampler.num_params).to(self.device) # use dimensionality of the full model

        elif not self.measure_rescaled:
            # set the limits for the integration bounds to the standard deviation of the laplace approximation times some number.
            if self.Rsampler.is_subspacelaplace:
                scaling = self.Rsampler.svd_S[:self.sampling_dims].sqrt() # use dimensionality of the subspace model
                # perhaps this works?
            else:
                scaling  = np.sqrt(torch.diag(self.Rsampler.covariance)) # use dimensionality of the full model
            self.limits = self.integral_bounds_std * scaling # use dimensionality of the sampling space (full or subspace)
            self.plot_limits = self.integral_bounds_std * torch.sqrt(torch.diag(self.Rsampler.covariance)[-self.sampling_dims:])  # use dimensionality of the subspace model
        if self.square_plots:
            self.plot_limits = self.plot_limits.max() * torch.ones_like(self.plot_limits).to(self.device)

        if self.measure_type=="gaussian":
            self.emukit_measure = GaussianMeasure(mean=np.zeros(self.sampling_dims), 
                                              variance=np.ones(self.sampling_dims))
            self.emukit_measure.reasonable_box = reasonable_box_fixed(self.emukit_measure, factor=self.integral_bounds_std)
            self.emukit_qrbf = QuadratureRBFGaussianMeasure(self.emukit_rbf, self.emukit_measure)
            self.plt_measure_dist = torch.distributions.MultivariateNormal(torch.zeros(self.sampling_dims), torch.eye(self.sampling_dims))

        if self.measure_type=="lebesgue":
            #print(f"{self.limits = }")
            #print(f"{self.plot_limits = }")
            limits_list_of_lists = [[-l, l] for l in self.limits]
            self.emukit_measure = LebesgueMeasure.from_bounds(limits_list_of_lists)
            self.emukit_qrbf = QuadratureRBFLebesgueMeasure(self.emukit_rbf, self.emukit_measure)
            self.plt_measure_dist = torch.distributions.Uniform(-self.plot_limits, self.plot_limits)
            self.plt_measure_dist.log_prob = lambda z: -torch.log((torch.tensor(self.plot_limits) * 2).prod()).repeat(z.shape[0])

        self.emukit_model = BaseGaussianProcessGPy(kern=self.emukit_qrbf, gpy_model=self.gpy_model)
        #self.emukit_method = VanillaBayesianQuadrature(base_gp=self.emukit_model, X=self.vs[0:1], Y=self.integrand_values[0:1])
        #print(f"{self.vs_ray.shape = }, {self.integrand_values.shape = }")
        self.emukit_method = VanillaBayesianQuadrature_multidim_output(base_gp=self.emukit_model, X=self.vs_ray.detach().cpu().numpy(), Y=self.integrand_values.detach().cpu().numpy())
        if use_ray_acqusition:
            self.ivr_acquisition = RayAcquisition(self.emukit_method, self.v_init.detach().cpu().numpy(), self.num_timesteps)
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
        v_new_t = torch.tensor(v_new).to(torch.float32).to(self.device)
        #print(f"{v_new = }")
        #print(f"{self.vs = }")
        self.vs = torch.cat([self.vs, v_new_t], dim=0)
        self.ys_new, thetas_new = self.integrand(v_new_t.squeeze(0))
        #print(f"{self.ys_new = }, {thetas_new = }")

        if self.use_rays:
            use_index = range(1, self.num_timesteps)
        else:
            use_index = [self.num_timesteps-1]
        #print(f"{use_index = }")
        #print(f"{thetas_new.shape = }")
        #print(f"{thetas_new[use_index].shape = }")
        self.thetas = torch.cat([self.thetas, thetas_new[use_index]], dim=0)
    
        spacing = torch.linspace(0, 1, self.num_timesteps, device=self.device).unsqueeze(-1)
        #vs_ray_new = np.linspace(self.v_init, v_new[0], self.num_timesteps)
        vs_ray_new = self.v_init + spacing * (v_new_t - self.v_init)
        self.vs_ray = torch.cat([self.vs_ray, vs_ray_new[use_index]], dim=0)
        
        self.integrand_values = torch.cat([self.integrand_values, self.ys_new[use_index]], dim=0)
        self.emukit_method.set_data(self.vs_ray.detach().cpu().numpy(), self.integrand_values.detach().cpu().numpy())
        integral_mean, integral_variance = self.emukit_method.integrate()
        self.integral_mean, self.integral_variance = torch.tensor(integral_mean, device=self.device, dtype=torch.float32), torch.tensor(integral_variance, device=self.device, dtype=torch.float32)
        self.steps += 1
        return self.integral_mean, self.integral_variance

    def lebesgue_integrand(self, v):
        num_ts = self.num_timesteps
        res, ts = self.Rsampler.expmap_torchdiffeq(self.Rsampler.mean, self.Rsampler.scale @ v, num_ts=num_ts)
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

    def predict_samples(self, xs):
        # returns a tensor with one row per parameter sample (theta) and then their correpsonding predictions for each x
        for i, theta in enumerate(self.thetas):
            params = vector_to_parameterdict(torch.tensor(theta, dtype=torch.float32), self.parametersubset)
            pred = self.functional_fwd_xs(params, xs)
            if i == 0:
                preds = torch.zeros((len(self.thetas), *pred.shape))
            preds[i] = pred
        return preds.permute(1,0,2)
    
    def pred_BQ_samples(self, xs, n_samples=100):
        preds = self.predict_samples(xs)
        X_bck = self.emukit_method.X # X here, is theta (for improved confusion / unreadability).
        Y_bck = self.emukit_method.Y

        # sample z's to evaluate GP in
        eps = torch.randn(n_samples, self.Rsampler.subspace_rank)

        # loop over each x and prediction(x)
        y_samples = torch.zeros((preds.shape[0], n_samples, preds.shape[2]), device=self.device)
        for i, pred in enumerate(preds):
            self.emukit_method.set_data(self.vs_ray.cpu().detach().numpy(), pred.cpu().detach().numpy()) # we have for each value of theta some values of pred(x)
            m, v = self.emukit_model.predict(eps.detach().numpy())
            y_samples[i] = torch.tensor(m, dtype=torch.float32, device=self.device)

        self.emukit_method.set_data(X_bck, Y_bck)
        return y_samples

    def pred_BQ(self, xs, get_mean=True, get_var=True, get_measure_variance=False, transform_type = "power", transform_param=2):
        preds = self.predict_samples(xs)
        X_bck = self.emukit_method.X # X here, is = z (for improved confusion / unreadability).
        Y_bck = self.emukit_method.Y

        # get means
        # loop over each x and prediction(x)
        for i, pred in enumerate(preds):
            self.emukit_method.set_data(self.vs_ray, pred) #we have for each value of theta some values of pred(x)
            mu, var = self.emukit_method.integrate() # integrate the function pred(theta given x)
            if i == 0:
                first_moments_mus = torch.zeros((len(preds), *mu.shape))
                first_moments_vars = torch.zeros((len(preds), *var.shape))
            first_moments_mus[i] = torch.as_tensor(mu)
            first_moments_vars[i] = torch.as_tensor(var)
        
        forward_transform, backwards_transform = transform(transform_type, param=transform_param)

        # get second moment
        for i, pred in enumerate(preds):
            sqr_error = (preds[i] -first_moments_mus.unsqueeze(1)[i])**2
            self.emukit_method.set_data(self.vs_ray, forward_transform(sqr_error))
            mu, var = self.emukit_method.integrate()
            #emukit_method = VanillaBayesianQuadrature(base_gp=self.emukit_model, X=self.vs_ray, Y=forward_transform(sqr_error))
            #mu, var = emukit_method.integrate()
            if i == 0:
                central_second_moment_mus = torch.zeros((len(preds), *mu.shape))
                central_second_moment_vars = torch.zeros((len(preds), *var.shape))
            trans = backwards_transform(torch.tensor(mu), torch.tensor(var))
            central_second_moment_mus[i] = trans[0]
            if len(trans) == 2: 
                central_second_moment_vars[i] = trans[1]
        
        self.emukit_method.set_data(X_bck, Y_bck)

        vars = [first_moments_vars]
        if len(trans)==2: vars += [central_second_moment_vars]

        returns = []
        if get_mean:
            returns += [first_moments_mus]
        if get_var:
            returns += [central_second_moment_mus]
        if get_measure_variance:
            returns += [vars]
        return returns

    def plot_integrand_2d(self, ax=None):
        # plot integrand values
        ax.scatter(self.vs_ray[:,0], self.vs_ray[:,1], c=self.integrand_values, cmap=self.contour_fill)
        ax.set_title("Integrand values")
        ax.set_xlim(-self.plot_limits[0], self.plot_limits[0])
        ax.set_ylim(-self.plot_limits[1], self.plot_limits[1])
        
    def plot_integrand_measure_2d(self, ax=None):
        ax.scatter(self.vs_ray[:,0], self.vs_ray[:,1], 
                    c=self.integrand_values[:,0] * self.plt_measure_dist.log_prob(torch.tensor(self.vs_ray)).exp().detach().numpy(), 
                    cmap=self.contour_fill)
        ax.set_title("Integrand values times measure")
        ax.set_xlim(-self.plot_limits[0], self.plot_limits[0])
        ax.set_ylim(-self.plot_limits[1], self.plot_limits[1])

    def plt_BQ_mean_2d(self, ax=None):
        self.plot_init_2d()
        mu_plot, var_plot = self.emukit_method.predict(self.xys_plt)
        ax.contourf(self.xs_plt, self.ys_plt, mu_plot.reshape(self.plot_N_mesh, self.plot_N_mesh).T, cmap=self.contour_fill)
        ax.contour(self.xs_plt, self.ys_plt, mu_plot.reshape(self.plot_N_mesh, self.plot_N_mesh).T, colors="black", linewidths=self.contour_linewidth)

        ax.scatter(self.vs_ray[:,0], self.vs_ray[:,1], c="k", marker="x")
        ax.set_title("BQ Model mean")
        ax.set_xlim(-self.plot_limits[0], self.plot_limits[0])
        ax.set_ylim(-self.plot_limits[1], self.plot_limits[1])
    
    def plot_BQ_mean_measure_2d(self, ax=None):
        self.plot_init_2d()
        mu_plot, var_plot = self.emukit_method.predict(self.xys_plt)
        ax.contourf(self.xs_plt, self.ys_plt, mu_plot.reshape(self.plot_N_mesh,self.plot_N_mesh).T * self.plt_measure_dist.log_prob(torch.tensor(self.xys_plt.reshape(self.plot_N_mesh, self.plot_N_mesh, -1))).exp().detach().numpy(), cmap=self.contour_fill)
        ax.contour(self.xs_plt, self.ys_plt, mu_plot.reshape(self.plot_N_mesh,self.plot_N_mesh).T * self.plt_measure_dist.log_prob(torch.tensor(self.xys_plt.reshape(self.plot_N_mesh, self.plot_N_mesh, -1))).exp().detach().numpy(), colors="black", linewidths=.1)
        
        ax.scatter(self.vs_ray[:,0], self.vs_ray[:,1], c="k", marker="x")
        ax.set_title("BQ Model mean times measure")

    def plot_BQ_std_2d(self, ax=None):
        self.plot_init_2d()
        mu_plot, var_plot = self.emukit_method.predict(self.xys_plt)
        ax.contourf(self.xs_plt, self.ys_plt, np.sqrt(var_plot).reshape(self.plot_N_mesh,self.plot_N_mesh).T, cmap=self.contour_fill)
        ax.contour(self.xs_plt, self.ys_plt, np.sqrt(var_plot).reshape(self.plot_N_mesh,self.plot_N_mesh).T, colors="black", linewidths=self.contour_linewidth)
        ax.scatter(self.vs_ray[:,0], self.vs_ray[:,1], c="k", marker="x")
        #axes[3].scatter(xys_plt[:,0], xys_plt[:,1], c=np.sqrt(var_plot), cmap="Wistia")
        ax.set_title("BQ Model std")

    def plot_acquisition_2d(self, ax=None):
        self.plot_init_2d()
        squared_correlation, integral_current_var, y_predictive_var, predictive_cov = self.ivr_acquisition._evaluate(self.xys_plt)
        ax.contourf(self.xs_plt, self.ys_plt, np.sqrt(squared_correlation).reshape(self.plot_N_mesh,self.plot_N_mesh).T, cmap=self.contour_fill)
        ax.contour(self.xs_plt, self.ys_plt, np.sqrt(squared_correlation).reshape(self.plot_N_mesh,self.plot_N_mesh).T, colors="black", linewidths=self.contour_linewidth)
        ax.scatter(self.vs_ray[:,0], self.vs_ray[:,1], c="k", marker="x")
        ax.set_title("Acquisition function")

    def plot_true_function_likelihood_2d(self, ax=None):
        self.plot_init_2d()
        self.get_funcvals()
        ax.contourf(self.xs_plt, self.ys_plt, self.function_times_likelihood.detach(), cmap=self.contour_fill)
        ax.scatter(self.thetas[:,0] - self.Rsampler.mean[0].detach().numpy(), self.thetas[:,1] - self.Rsampler.mean[1].detach().numpy(), c="k", marker="x")
        
        rays = self.thetas[1:].reshape(-1, self.num_timesteps-1, 2).permute(0, 2, 1)
        rays2 = np.concatenate([np.tile(self.thetas[None, 0, None].permute(0, 2, 1), [rays.shape[0],1,1]), rays] , axis=2)

        for i, sample in enumerate(rays2):
            ax.plot(sample[0]-self.Rsampler.mean[0].detach().numpy(), 
                    sample[1]-self.Rsampler.mean[1].detach().numpy(), marker=None, c="k", alpha=0.3)

        ax.set_title("True function times model likelihood")
        ax.set_xlim(-self.plot_limits[0], self.plot_limits[0])
        ax.set_ylim(-self.plot_limits[1], self.plot_limits[1])
        if self.theta_space_plot_limits is not None:
            ax.set_xlim(self.theta_space_plot_limits[0])
            ax.set_ylim(self.theta_space_plot_limits[1])


    def plot_model_likelihood_2d(self, ax=None, function_vals = True):
        self.plot_init_2d()
        self.get_funcvals()
        ax.contourf(self.xs_plt, self.ys_plt, (-self.plt_model_loss*2).exp().detach(), cmap=self.contour_fill)
        ax.contour(self.xs_plt, self.ys_plt, (-self.plt_model_loss*2).exp().detach(), colors="black", linewidths=self.contour_linewidth)
        
        if function_vals:
            ax.contour(self.xs_plt, self.ys_plt, self.function_vals.detach(), cmap="Grays")
        
        ax.scatter(self.thetas[:,0] - self.Rsampler.mean[0].detach().numpy(), self.thetas[:,1] - self.Rsampler.mean[1].detach().numpy(), c="k", marker="x")

        rays = self.thetas[1:].reshape(-1, self.num_timesteps-1, 2).permute(0, 2, 1)
        rays2 = np.concatenate([np.tile(self.thetas[None, 0, None].permute(0, 2, 1), [rays.shape[0],1,1]), rays] , axis=2)

        for i, sample in enumerate(rays2):
            ax.plot(sample[0]-self.Rsampler.mean[0].detach().numpy(), 
                    sample[1]-self.Rsampler.mean[1].detach().numpy(), marker=None, c="k", alpha=0.3)


        #axes[0][3].contour(self.xs_plt, self.ys_plt, self.function_vals.detach(), c="k")
        ax.set_title("Model loss and true function values")
        ax.set_xlim(-self.plot_limits[0], self.plot_limits[0])
        ax.set_ylim(-self.plot_limits[1], self.plot_limits[1])
        if self.theta_space_plot_limits is not None:
            ax.set_xlim(self.theta_space_plot_limits[0])
            ax.set_ylim(self.theta_space_plot_limits[1])


    def plot_init_2d(self):
        if not self.has_plotted:
            self.xs_plt = np.linspace(-self.plot_limits[0], self.plot_limits[0], self.plot_N_mesh)
            self.ys_plt = np.linspace(-self.plot_limits[1], self.plot_limits[1], self.plot_N_mesh)
            self.xys_plt = np.array([[x, y] for x in self.xs_plt for y in self.ys_plt])
    
    def get_funcvals(self):
        if not self.has_plotted:
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
        ax.scatter(self.vs_ray[:,0], self.integrand_values[:,0] * plt_measure_points , c="k", marker="x", label="Integrand times measure")
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
        ax.scatter(self.vs_ray[:,0], self.integrand_values, c="k", marker="x", label="Integrand")
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

        ax.scatter(self.thetas[:,0] - self.Rsampler.mean[0].detach().numpy(), self.integrand_values, c="k", marker="x", label="Integrand")
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
        self.function_vals = torch.stack([self.functional_fwd(obs + self.Rsampler.mean) for obs in torch.tensor(self.xs_plt_1d, dtype=torch.float32)]).reshape(self.plot_N_mesh)
        self.plt_model_loss = torch.stack([self.Rsampler.f_loss(obs + self.Rsampler.mean) for obs in torch.tensor(self.xs_plt_1d, dtype=torch.float32)])
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
            fig, axes = plt.subplots(2,3, figsize=(15,10))
            self.plot_integrand_2d(ax=axes[0][0])
            self.plot_integrand_measure_2d(ax=axes[0][1])
            self.plt_BQ_mean_2d(ax=axes[1][0])
            self.plot_BQ_mean_measure_2d(ax=axes[1][1])
            self.plot_BQ_std_2d(ax=axes[1][2])
            self.plot_acquisition_2d(ax=axes[0][2])
            self.has_plotted = True
            return fig, axes


        elif self.Rsampler.num_params == 2 and self.sampling_dims == 1: # This is the case where we have 2d parameters and 1d subspace
            fig, axes = plt.subplots(1,4, figsize=(15,5))
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

