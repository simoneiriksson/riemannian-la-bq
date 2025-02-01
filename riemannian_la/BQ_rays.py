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
                 integral_bounds_std=2, GP_lengthscale=1.0, GP_variance=1.0, num_timesteps=10, use_ray_acqusition=True, use_rays=True
                 ):
        self.Rsampler = Rsampler
        self.evaluation_model = evaluation_model
        self.measure = measure
        self.integral_bounds_std = integral_bounds_std
        self.GP_lengthscale = GP_lengthscale
        self.GP_variance = GP_variance

        self.functional_fwd = make_functional_fwd_vector(evaluation_model, xs, parametersubset=parametersubset)

        self.num_timesteps = num_timesteps


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

        self.v_init = np.zeros((self.Rsampler.num_params))
        self.y_init, self.theta_init = self.integrand(torch.tensor(self.v_init, dtype=torch.float32))
        self.y_init, self.theta_init = self.y_init[0,:].detach().numpy(), self.theta_init[0,:].detach().numpy()   
        
        self.vs = np.expand_dims(self.v_init, 0)
        self.vs_ray = self.vs
        self.model_output = np.expand_dims(self.y_init, 0)
        self.thetas = np.expand_dims(self.theta_init, 0)
        
    # GPy takes X and Y values at initialization. Those will be overwritten later when the emukit model is initialized.
        self.BQ_kernel = kernel=GPy.kern.RBF(
                                    input_dim=self.Rsampler.num_params, 
                                    lengthscale=self.GP_lengthscale,
                                    variance=self.GP_variance)
        #print(f"{self.vs.shape = }, {self.model_output.shape = }")
        self.gpy_model = GPy.models.GPRegression(X=self.vs[0:1], 
                                                 Y=self.model_output[0:1], 
                                                 kernel=self.BQ_kernel)
        self.emukit_rbf = RBFGPy(self.gpy_model.kern)

        if self.measure_rescaled:
            self.limits = self.integral_bounds_std * np.ones(self.Rsampler.num_params)
        elif not self.measure_rescaled:
            self.limits = self.integral_bounds_std * np.sqrt(torch.diag(self.Rsampler.covariance).detach().numpy())

        if self.measure_type=="gaussian":
            self.emukit_measure = GaussianMeasure(mean=np.zeros(self.Rsampler.num_params), 
                                              variance=np.ones(self.Rsampler.num_params))
            self.emukit_measure.reasonable_box = reasonable_box_fixed(self.emukit_measure, factor=self.integral_bounds_std)
            self.emukit_qrbf = QuadratureRBFGaussianMeasure(self.emukit_rbf, self.emukit_measure)

        if self.measure_type=="lebesgue":
            limits_list_of_lists = [[-l, l] for l in self.limits]
            self.emukit_measure = LebesgueMeasure.from_bounds(limits_list_of_lists)
            self.emukit_qrbf = QuadratureRBFLebesgueMeasure(self.emukit_rbf, self.emukit_measure)

        self.emukit_model = BaseGaussianProcessGPy(kern=self.emukit_qrbf, gpy_model=self.gpy_model)
        self.emukit_method = VanillaBayesianQuadrature(base_gp=self.emukit_model, X=self.vs[0:1], Y=self.model_output[0:1])
        if use_ray_acqusition:
            self.ivr_acquisition = RayAcquisition(self.emukit_method, self.v_init, self.num_timesteps)
        else:
            self.ivr_acquisition = IntegralVarianceReduction(self.emukit_method)

        self.space = ParameterSpace(self.emukit_method.reasonable_box_bounds.convert_to_list_of_continuous_parameters())
        self.optimizer = GradientAcquisitionOptimizer(self.space)
        self.has_plotted = False
        self.use_rays = use_rays

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
        
        self.model_output = np.append(self.model_output, self.ys_new[use_index].detach().numpy(), axis=0)
        self.emukit_method.set_data(self.vs_ray, self.model_output)
        self.integral_mean, self.integral_variance = self.emukit_method.integrate()
        return self.integral_mean, self.integral_variance

    def lebesgue_integrand(self, v):
        num_ts = self.num_timesteps
        res, ts = self.Rsampler.expmap_torchdiffeq(self.Rsampler.mean, v, num_ts=num_ts)
        #theta = res[-1,:self.Rsampler.num_params].T
        thetas = res[:,:self.Rsampler.num_params]
        vs_ray = torch.tensor(np.linspace(self.v_init, v.detach().numpy(), self.num_timesteps))
        v_dist = torch.distributions.MultivariateNormal(torch.zeros_like(self.Rsampler.mean), self.Rsampler.covariance)
        #ys = torch.stack([self.functional_fwd(theta) * self.Rsampler.posterior_logprob(vs_ray[i]).exp() for i, theta in enumerate(thetas)])
        #print(f"{vs_ray = }, {thetas = }")
        ys = torch.stack([self.functional_fwd(theta) * v_dist.log_prob(vs_ray[i]).exp() for i, theta in enumerate(thetas)])
        #return self.functional_fwd(theta) * R_sampler.posterior_logprob(v).exp(), theta
        return ys, thetas

    def lebesgue_integrand_rescale(self, z):
        num_ts = self.num_timesteps
        g = lambda z: self.Rsampler.scale @ z
        g_inv = lambda v: self.Rsampler.scale.inverse() @ v
        v = g(z)
        res, ts = self.Rsampler.expmap_torchdiffeq(self.Rsampler.mean, v, num_ts=num_ts)
        #vs_ray = torch.tensor(np.linspace(self.v_init, v.detach().numpy(), self.num_timesteps))
        z_init = g_inv(torch.tensor(self.v_init, dtype=torch.float32)).detach().numpy()
        zs_ray = torch.tensor(np.linspace(z_init, z.detach().numpy(), self.num_timesteps))
        #print(f"{vs_ray.shape = }, {zs_ray.shape = }")
        #zs_ray = torch.stack([g_inv(v) for v in vs_ray])
        #theta = res[-1,:self.Rsampler.num_params].T
        thetas = res[:,:self.Rsampler.num_params]
        z_dist = torch.distributions.MultivariateNormal(torch.zeros_like(self.Rsampler.mean), torch.eye(self.Rsampler.covariance.shape[0]))
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

    def plot2d(self):
        if self.Rsampler.num_params != 2:
            raise ValueError("Can only plot 1d")
        else:
            N=100
            if not self.has_plotted:
                self.xs_plt = np.linspace(-self.limits[0], self.limits[0], N)
                self.ys_plt = np.linspace(-self.limits[1], self.limits[1], N)
                self.xys_plt = np.array([[x, y] for x in self.xs_plt for y in self.ys_plt])
            fig, axes = plt.subplots(2,4, figsize=(20,10))

            mu_plot, var_plot = self.emukit_method.predict(self.xys_plt)
            squared_correlation, integral_current_var, y_predictive_var, predictive_cov = self.ivr_acquisition._evaluate(self.xys_plt)
            if self.measure_type == "gaussian":
                z_dist = torch.distributions.MultivariateNormal(torch.zeros_like(self.Rsampler.mean), torch.eye(self.Rsampler.covariance.shape[0]))
            elif self.measure_type == "lebesgue":
                z_dist = torch.distributions.Uniform(torch.tensor(-self.limits), torch.tensor(self.limits))
                z_dist.log_prob = lambda z: -torch.log((torch.tensor(self.limits) * 2).prod())

            axes[0][0].scatter(self.vs_ray[:,0], self.vs_ray[:,1], c=self.model_output, cmap="viridis")
            axes[0][0].set_title("Function values")
            axes[0][0].set_xlim(-self.limits[0], self.limits[0])
            axes[0][0].set_ylim(-self.limits[1], self.limits[1])

            axes[0][1].scatter(self.vs_ray[:,0], self.vs_ray[:,1], c=self.model_output[:,0] * z_dist.log_prob(torch.tensor(self.vs_ray)).exp().detach().numpy(), cmap="viridis")
            axes[0][1].set_title("Function times measure")
            axes[0][1].set_xlim(-self.limits[0], self.limits[0])
            axes[0][1].set_ylim(-self.limits[1], self.limits[1])

            axes[1][0].contourf(self.xs_plt, self.ys_plt, mu_plot.reshape(N,N).T, cmap="viridis")
            axes[1][0].scatter(self.vs_ray[:,0], self.vs_ray[:,1], c="k", marker="x")
            axes[1][0].set_title("BQ Model mean")

            axes[1][1].contourf(self.xs_plt, self.ys_plt, mu_plot.reshape(N,N).T * z_dist.log_prob(torch.tensor(self.xys_plt.reshape(N, N, -1))).exp().detach().numpy(), cmap="viridis")
            axes[1][1].scatter(self.vs_ray[:,0], self.vs_ray[:,1], c="k", marker="x")
            axes[1][1].set_title("BQ Model mean times measure")

            axes[1][2].contourf(self.xs_plt, self.ys_plt, np.sqrt(var_plot).reshape(N,N).T, cmap="viridis")
            axes[1][2].scatter(self.vs_ray[:,0], self.vs_ray[:,1], c="k", marker="x")
            #axes[3].scatter(xys_plt[:,0], xys_plt[:,1], c=np.sqrt(var_plot), cmap="viridis")
            axes[1][2].set_title("BQ Model std")

            axes[1][3].contourf(self.xs_plt, self.ys_plt, np.sqrt(squared_correlation).reshape(N,N).T, cmap="viridis")
            axes[1][3].scatter(self.vs_ray[:,0], self.vs_ray[:,1], c="k", marker="x")
            #axes[3].scatter(xys_plt[:,0], xys_plt[:,1], c=np.sqrt(var_plot), cmap="viridis")
            axes[1][3].set_title("Acquisition function")





            self.has_plotted = True
            return fig, axes







#################
# Now for 2d test
plot=True
n_mesh = 1000
curvature = 0.1
banana_function = functional_banana(curvature=curvature, sigma_x=2.0, sigma_y=1.0)

displace = 1.0
banana_function_moved = lambda x: banana_function(x+displace)
xs = torch.tensor([-1.0, -1.0]).unsqueeze(0)*displace
ys = banana_function_moved(xs[0]).unsqueeze(0)

class tiny_ridiculess_model_class(torch.nn.Module):
    def __init__(self, n_params):
        super().__init__()
        self.params = torch.nn.Parameter(torch.arange(n_params).float())
    def forward(self, x):
        #return torch.sin(self.params*3.1).sum().repeat(x.shape[0], 1)**2
        #return torch.ones(x.shape[0], 1)
        return (self.params+displace).sum().repeat(x.shape[0], 1)**2
        #return self.params.sum().repeat(x.shape[0], 1)+2
        
evaluation_model = tiny_ridiculess_model_class(n_params=2)
functional_evaluation_model = make_functional_fwd_xs(evaluation_model)  # make functional version of model

banana_model = Model_from_func(banana_function_moved, input_shape=[2])
torch.nn.utils.vector_to_parameters(xs[0], banana_model.parameters())
parametersubset = dict(banana_model.named_parameters())

R_sampler = Riemann_sampler(banana_model, xs=xs, ys=ys, loss_fn=neglog_loss(), prior_sigma=0)
R_sampler.fit(fitting_type="hessian")

evaluation_model = tiny_ridiculess_model_class(n_params=2)

torch.manual_seed(0)
BQ = BayesianQuadrature_rays(R_sampler, evaluation_model, measure="lebesgue", integral_bounds_std=4, 
                             GP_lengthscale=1.0, GP_variance=1.0, num_timesteps=10, use_ray_acqusition=True, use_rays=True)
for i in range(100):
    integral_mean, integral_variance = BQ.step()
    print(f"{i = }, {integral_mean = }, {integral_variance = }")
    if i%1 ==0:
        fig, axes = BQ.plot2d()
        plt.show()


def loop(measure, num_steps=10):
    BQ = BayesianQuadrature_rays(R_sampler, evaluation_model, measure=measure, integral_bounds_std=4, 
                                 GP_lengthscale=1.0, GP_variance=1.0, num_timesteps=10, use_ray_acqusition=True, use_rays=True)
    for i in range(num_steps):
        integral_mean, integral_variance = BQ.step()
        print(f"{i = }, {integral_mean = }, {integral_variance = }")
        fig, axes = BQ.plot2d()
        plt.show()


num_steps =20
loop("gaussian_rescaled", num_steps=num_steps)
loop("lebesgue_rescaled", num_steps=num_steps)
loop("lebesgue", num_steps=num_steps)










torch.manual_seed(0)
BQ_noray = BayesianQuadrature_rays(R_sampler, evaluation_model, measure="lebesgue", integral_bounds_std=4, 
                             GP_lengthscale=1.0, GP_variance=1.0, num_timesteps=10, use_ray_acqusition=True, use_rays=True)
plt.show()
for i in range(100):
    integral_mean, integral_variance = BQ_noray.step()
    print(f"{i = }, {integral_mean = }, {integral_variance = }")
    if i%10 ==0:
        fig, axes = BQ_noray.plot2d()
        plt.show()



eval_in_vs_mean1, eval_in_vs_var1 = BQ_noray.emukit_method.base_gp.predict(BQ_ray.vs_ray)
eval_in_vs_mean2, eval_in_vs_var2 = BQ_ray.emukit_method.base_gp.predict(BQ_ray.vs_ray)

eval_in_vs_mean3, eval_in_vs_var3 = BQ_noray.emukit_method.base_gp.predict(BQ_noray.vs_ray)
eval_in_vs_mean4, eval_in_vs_var4 = BQ_ray.emukit_method.base_gp.predict(BQ_noray.vs_ray)


max_obs_ray = BQ_ray.model_output.argmax()
max_obs_ray = 1
print(f"{max_obs_ray = }")
print(f"{BQ_ray.model_output[max_obs_ray] = }") 
print(f"{BQ_ray.emukit_method.base_gp.predict(BQ_ray.vs_ray[max_obs_ray:max_obs_ray+1]) = }") 

print(f"{BQ_ray.vs_ray[max_obs_ray] = }")
print(f"{BQ_ray.thetas[max_obs_ray] = }")

#print(f"{BQ_noray.integrand(torch.tensor(BQ_ray.vs_ray[max_obs_ray], dtype=torch.float32)) = }")
print(f"{BQ_ray.integrand(torch.tensor(BQ_ray.vs_ray[max_obs_ray], dtype=torch.float32)) = }")

max_obs_noray = BQ_noray.model_output.argmax()
print(f"{max_obs_noray = }")
print(f"{BQ_noray.model_output[max_obs_noray] = }")
print(f"{BQ_noray.vs_ray[max_obs_noray] = }")
print(f"{BQ_noray.thetas[max_obs_noray] = }")

print(f"{BQ_noray.integrand(torch.tensor(BQ_noray.vs_ray[max_obs_noray], dtype=torch.float32)) = }")
print(f"{BQ_ray.integrand(torch.tensor(BQ_noray.vs_ray[max_obs_noray], dtype=torch.float32)) = }")


torch.manual_seed(0)
BQ_ray_copy = BayesianQuadrature_rays(R_sampler, evaluation_model, measure="lebesgue_rescaled", integral_bounds_std=4, 
                             GP_lengthscale=1.0, GP_variance=1.0, num_timesteps=5, use_ray_acqusition=True, use_rays=True)

BQ_ray_copy.emukit_method.set_data(BQ_ray.vs_ray, BQ_ray.model_output)
BQ_ray_copy.vs_ray = BQ_ray.vs_ray
BQ_ray_copy.model_output = BQ_ray.model_output
BQ_ray_copy.thetas = BQ_ray.thetas
BQ_ray_copy.vs = BQ_ray.vs

_= BQ_ray_copy.plot2d()
print(f"{BQ_ray_copy.emukit_method.integrate() = }")

print(f"{BQ_ray_copy.integrand(torch.tensor(BQ_ray.vs_ray[max_obs_ray], dtype=torch.float32)) = }")


BQ_ray_copy.vs_ray = np.append(BQ_ray.vs_ray, BQ_noray.vs_ray, axis=0)
BQ_ray_copy.model_output = np.append(BQ_ray.model_output, BQ_noray.model_output, axis=0)
BQ_ray_copy.thetas = np.append(BQ_ray.thetas, BQ_noray.thetas, axis=0)
BQ_ray_copy.vs = np.append(BQ_ray.vs, BQ_noray.vs_ray, axis=0)

BQ_ray_copy.emukit_method.set_data(BQ_ray_copy.vs_ray, BQ_ray_copy.model_output)

_= BQ_ray_copy.plot2d()
print(f"{BQ_ray_copy.emukit_method.integrate() = }")
max_obs_copy = BQ_ray_copy.model_output.argmax()
print(f"{max_obs_copy = }")
print(f"{BQ_ray_copy.model_output[max_obs_copy] = }")
print(f"{BQ_ray_copy.vs_ray[max_obs_copy] = }")
print(f"{BQ_ray_copy.thetas[max_obs_copy] = }")
print(f"{BQ_ray_copy.integrand(torch.tensor(BQ_ray_copy.vs_ray[max_obs_copy], dtype=torch.float32)) = }")

# find closest point to max_obs_ray in BQ_noray
distances = np.linalg.norm(np.delete(BQ_ray_copy.vs_ray, max_obs_copy, 0) - BQ_ray_copy.vs_ray[max_obs_copy], axis=1)
min_index = distances.argmin()
print(f"{min_index = }")
print(f"{distances[min_index] = }")
print(f"{BQ_ray_copy.model_output[min_index] = }")
print(f"{BQ_ray_copy.vs_ray[min_index] = }")
print(f"{BQ_ray_copy.thetas[min_index] = }")
print(f"{BQ_ray_copy.integrand(torch.tensor(BQ_ray_copy.vs_ray[min_index], dtype=torch.float32)) = }")




eval_in_vs_mean, eval_in_vs_var = BQ_ray.emukit_method.base_gp.predict(BQ_ray.vs_ray[28])


vs_ray = BQ.vs_ray
model_output = BQ.model_output
eval_in_vs_mean, eval_in_vs_var = BQ.emukit_method.base_gp.predict(vs_ray)

plt.scatter(x=model_output[:,0], y=eval_in_vs_mean[:,0])
model_output[:,0]- eval_in_vs_mean[:,0]




vs_ray_2 = BQ.vs_ray
vs_ray.shape
model_output_2 = BQ.model_output
eval_in_vs_mean_2, eval_in_vs_var_2 = BQ.emukit_method.base_gp.predict(vs_ray_2)

eval_in_vs_mean_2-eval_in_vs_mean

