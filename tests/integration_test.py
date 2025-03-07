import os
import sys
# set working directory
print(f"{__file__ = }")
print(f"{sys.path = }")
print("working dir:", os.getcwd())
os.chdir("../riemannian_la")
print("working dir:", os.getcwd())
from models import LinearModel, Model_from_func
from getdata import gen_log_regression_data
from train import train
from laplace_approx import Laplace, vector_to_parameterdict
from utils import loss_func_from_target_sigma, make_functional_fwd_xs, functional_loss, functional_loss_for_vmap, sum_loss, neglog_loss
from discrete_sampler import discrete_function_sampler, discrete_model_sampler
from riemann_sampler import Riemann_sampler, riemann_plotter
from integration import integrator
from torch.func import grad, jvp, vjp, hessian, jacfwd, jacrev, vmap, functional_call
from matplotlib import pyplot as plt
from laplace_approx import Laplace
from MCMC_sampler import MCMC_sampler
import torch
from models import functional_banana
from MCMC_sampler import MCMC_sampler
import seaborn as sns
import pandas as pd
from BQ_rays_subspace import BayesianQuadrature_rays

fig_folder = "../report/figures"
os.makedirs(fig_folder, exist_ok=True)
torch.manual_seed(0)
"""
A series of usage tests for the integrator function
Banana model:
1)  discrete integration over function:
    We want to take the expectation of the output of tiny_ridiculess_model_class (which is equal to x^2 + y^2) over the parameterspace, restricted to some box.
    Here, the paramters folow the posterior distribution given (exactly) by the banana FUNCTION

2)  discrete integration over model:
    We want to take the expectation of the output of tiny_ridiculess_model_class (which is equal to x^2 + y^2) over the parameterspace, restricted to some box.
    Here, the paramters folow the posterior distribution given (exactly) by the banana MODEL

3)  laplace integration over model:
    We want to take the expectation of the output of tiny_ridiculess_model_class (which is equal to x^2 + y^2) over the parameterspace, restricted to some box.
    Here, the paramters folow the laplace approximation of the posterior distribution given by the banana MODEL

4)  MCMC-integration over banana model posterior:

Logistic regression model:
5)  

"""

n_mesh = 200
curvature = 0.25
span = 20
plot_xlim=(-4,4)
plot_ylim=(-1,4)


banana_function = functional_banana(curvature=curvature, sigma_x=2.0, sigma_y=.5)
xs = torch.tensor([0.0, 0.0]).unsqueeze(0)
ys = banana_function(xs[0]).unsqueeze(0)
class tiny_ridiculess_model_class(torch.nn.Module):
    def __init__(self, n_params):
        super().__init__()
        self.params = torch.nn.Parameter(torch.arange(n_params).float())
    def forward(self, x):
        #return torch.sin(self.params*3.1).sum().repeat(x.shape[0], 1)**2
        #return torch.ones(x.shape[0], 1)
        #return self.params.sum().repeat(x.shape[0], 1)**2
        return self.params.sum().repeat(x.shape[0], 1)

const_prior = lambda x: torch.tensor(0.0)


evaluation_model = tiny_ridiculess_model_class(n_params=2)
functional_evaluation_model = make_functional_fwd_xs(evaluation_model)  # make functional version of model
limits = [[-span, span], [-span, span]]


####################################
# 1) discrete integration over function
####################################
#Let's do discrete integration with the banana function
#banana_function = functional_banana(curvature=2.0, sigma_x=2.0, sigma_y=1.0)
#neg_banana = lambda x: -functional_banana(curvature=0.0, sigma_x=1.0, sigma_y=1.0)(x)
discrete_sampler_fn = discrete_function_sampler(func=banana_function, limits=limits, n_mesh=n_mesh, 
                                             normalize_weights=False)
posterior_samples, weights = discrete_sampler_fn.samples_and_weights()

banana_function(posterior_samples[0])
sampler = discrete_sampler_fn

parametersubset = dict(evaluation_model.named_parameters())

integral_discrete, function_values, weights, posterior_samples = integrator(sampler, functional_evaluation_model, parametersubset, xs)
print(f"1) When using discrete integration over banana FUNCTION we get {integral_discrete = }")

plt.scatter(posterior_samples[:,0], posterior_samples[:,1], c=weights)
plt.colorbar()
plt.show()
plt.scatter(posterior_samples[:,0], posterior_samples[:,1], c=function_values)
plt.show()


####################################
# 2) discrete integration over model
####################################
banana_function_ = lambda x: banana_function(x)
banana_model = Model_from_func(banana_function_, input_shape=[2])
parametersubset = dict(banana_model.named_parameters())
x = torch.tensor([0.0, 0.0])
#print(f"{banana_function(x) = }")
torch.nn.utils.vector_to_parameters(torch.tensor([0.0, 0.0]), banana_model.parameters())
#print(f"{banana_model(x) = }")

xs = torch.tensor([0.0, 0.0]).unsqueeze(0)
ys = banana_function_(xs[0]).unsqueeze(0)

discrete_sampler = discrete_model_sampler(banana_model, loss_fn=neglog_loss(), xs=xs, ys=ys, 
                                          limits=[[-span, span], [-span, span]], n_mesh=n_mesh, normalize_weights=True, prior_loss=const_prior)
posterior_samples, weights = discrete_sampler.samples_and_weights()
integral_discrete, function_values, weights, posterior_samples = integrator(discrete_sampler, functional_evaluation_model, parametersubset, xs)
print(f"When using discrete integration over bnana MODEL we get {integral_discrete = }")
plt.scatter(posterior_samples[:,0], posterior_samples[:,1], c=weights)
plt.colorbar()
plt.show()
plt.close()

fig, ax =plt.subplots()
ax.contour(posterior_samples[:,0].reshape(n_mesh, n_mesh).detach().numpy(), 
             posterior_samples[:,1].reshape(n_mesh, n_mesh).detach().numpy(), 
             weights.reshape(n_mesh, n_mesh).detach().numpy(), levels=10, colors="black")
ax.set_xlim(plot_xlim)
ax.set_ylim(plot_ylim)

plt.savefig(os.path.join(fig_folder, "banana_dist_discrete_isocurves.png"))
plt.close()

####################################
# 3) laplace integration over model
####################################

# and now the laplace approximation with the banana function
banana_model = Model_from_func(banana_function, input_shape=[2])
torch.nn.utils.vector_to_parameters(torch.tensor([0.0, 0.0]), banana_model.parameters())
dict(banana_model.named_parameters())
const_prior = lambda x: torch.tensor(0.0)

laplace = Laplace(banana_model, xs=xs, ys=ys, loss_fn=neglog_loss(), prior_loss=const_prior)
laplace.fit(fitting_type="hessian", xs=xs, ys=ys)

for i in range(10):
  laplace.make_posterior_sample(n_samples=10000)
  integral_la, function_values_lp, weights_lp, posterior_samples_la = integrator(laplace, functional_evaluation_model, parametersubset, xs)
  laplace.posterior_samples.T.cov()
  print(f"When using the laplace approxiation of posterior of banana MODEL with {laplace.posterior_samples.shape[0]} we get {integral_la = }")

plt.scatter(posterior_samples_la[:,0].detach(), posterior_samples_la[:,1].detach(), c=weights_lp.detach())
plt.colorbar()
plt.show()
plt.close()

df = pd.DataFrame(posterior_samples_la.detach().numpy())
fig, ax =plt.subplots()
sns.kdeplot(df, x=0, y=1, kind="kde", levels=10, color="black", ax=ax)
ax.set_xlim(plot_xlim)
ax.set_ylim(plot_ylim)
plt.savefig(os.path.join(fig_folder, "banana_dist_laplace_isocurves.png"))
plt.show()
plt.close()
#plt.scatter(posterior_samples_la[:,0].detach(), posterior_samples_la[:,1].detach(), c=function_values_lp.detach())


#######################################
# 4) MCMC-integration from banana model
#######################################


# and now the laplace approximation with the banana function
banana_model = Model_from_func(banana_function, input_shape=[2])
torch.nn.utils.vector_to_parameters(torch.tensor([0.0, 0.0]), banana_model.parameters())
#loss_fn = lambda preds, target: torch.sum(preds.log())

parametersubset = dict(banana_model.named_parameters())
sampler_mcmc = MCMC_sampler(banana_model, parametersubset, xs=xs, ys=ys, loss_fn=neglog_loss(), prior_loss=const_prior)
_=sampler_mcmc.make_posterior_sample(10000)
sampler_mcmc.posterior_samples.T.cov()

integral_mcmc, function_values_mcmc, weights_mcmc, posterior_samples_mcmc = integrator(sampler_mcmc, functional_evaluation_model, parametersubset, xs)

print(f"When using the MCMC sampling of posterior of banana MODEL with {sampler_mcmc.posterior_samples.shape[0]} samples we get {integral_mcmc = }")

plt.scatter(posterior_samples_mcmc[:,0].detach(), posterior_samples_mcmc[:,1].detach(), c=function_values_mcmc.detach())
plt.colorbar()
plt.show()
plt.close()

df = pd.DataFrame(posterior_samples_mcmc.detach().numpy(), columns=["x", "y"])
fig, ax =plt.subplots()
sns.kdeplot(df, x="x", y="y", kind="kde", levels=10, color="black", ax=ax)
ax.set_xlim(plot_xlim)
ax.set_ylim(plot_ylim)
plt.savefig(os.path.join(fig_folder, "banana_dist_mcmc_isocurves.png"))
plt.close()

####################################
# 5) Riemannian laplace integration over model
####################################

# and now the laplace approximation with the banana function
banana_model = Model_from_func(banana_function, input_shape=[2])
torch.nn.utils.vector_to_parameters(torch.tensor([0.0, 0.0]), banana_model.parameters())
parametersubset = dict(banana_model.named_parameters())

R_sampler = Riemann_sampler(banana_model, parametersubset, xs=xs, ys=ys, loss_fn=neglog_loss(), prior_loss=const_prior, subspace_rank=2)
R_sampler.fit(fitting_type="hessian")

integral_vals_riemann = []
n_samples_riemann = []

BQ = BayesianQuadrature_rays(R_sampler, evaluation_model=evaluation_model, measure="gaussian_rescaled", integral_bounds_std=4, 
                             GP_lengthscale=1.0, GP_variance=1.0, num_timesteps=7, use_ray_acqusition=True,  
                             use_rays=True, 
                             theta_space_plot_limits=[[-2,2], [-2,2]], xs = xs[0], parametersubset=None)

for i in range(10):
  #_=R_sampler.make_posterior_sample_la(1)
  R_params = R_sampler.make_posterior_sample(10)
  integral_riemann, function_values_riemann, weights_riemann, posterior_samples_riemann = integrator(R_sampler, functional_evaluation_model, parametersubset, xs)
  print(f"When using Riemannian Laplace approximation with {R_sampler.posterior_samples_la.shape[0]} samples we get {integral_riemann = }")
  integral_vals_riemann += [integral_riemann[0,0]]
  n_samples_riemann += [R_sampler.posterior_samples_la.shape[0]]

  # BQ.emukit_model.set_data(X=R_sampler.posterior_samples_la.detach(), Y=function_values_riemann[:,:,0])
  # integral_mean, integral_variance = BQ.emukit_method.integrate()
  # print(f"With added BQ using Riemannian Laplace approximation with {R_sampler.posterior_samples_la.shape[0]} samples we get {integral_mean = }")

#fig, ax = riemann_plotter(R_sampler, sample_markers=".", plot_traject=True, plot_traj_marker=None, max_samples=None, LA_arrows=[1])

df = pd.DataFrame(R_sampler.posterior_samples.numpy(), columns=["x", "y"])
fig, ax =plt.subplots()
sns.kdeplot(data=df, x="x", y="y", levels=10, color="black", ax=ax)
ax.set_xlim(plot_xlim)
ax.set_ylim(plot_ylim)
plt.show()
plt.savefig(os.path.join(fig_folder, "banana_dist_riemann_isocurves.png"))
plt.close()

plt.plot(n_samples_riemann, integral_vals_riemann)
plt.savefig(os.path.join(fig_folder, "banana_dist_riemann_convergence.png"))
plt.close()


####################################
# 6) Bayesian Quadrature - Riemannian laplace integration over model
####################################
R_sampler_2d = Riemann_sampler(banana_model, xs=xs, ys=ys, loss_fn=neglog_loss(), prior_loss=const_prior, subspace_rank=2)
_=R_sampler_2d.fit(fitting_type="hessian")

BQ_2d = BayesianQuadrature_rays(R_sampler_2d, evaluation_model=evaluation_model, measure="gaussian_rescaled", integral_bounds_std=4, 
                             GP_lengthscale=1.0, GP_variance=1.0, num_timesteps=7, use_ray_acqusition=True,  
                             use_rays=True, 
                             theta_space_plot_limits=[[-2,2], [-2,2]], xs = xs[0], parametersubset=None)
BQ_2d.theta_space_plot_limits=[[-4,4], [-4,4]]

integral_vals_riemann_BQ = []
n_samples_riemann_BQ = []
n_steps= []

for i in range(16):
    integral_mean_BQ, integral_variance_BQ = BQ_2d.step()
    print(f"{BQ_2d.steps = }, observations = {BQ_2d.emukit_method.X.shape[0]}, {integral_mean_BQ = }, {integral_variance_BQ = }")
    integral_vals_riemann_BQ += [integral_mean_BQ]
    n_samples_riemann_BQ += [BQ_2d.emukit_method.X.shape[0]]
    n_steps += [BQ_2d.steps]
    if i == 7: 
        fig, axes = BQ_2d.plot()
        plt.savefig(os.path.join(fig_folder, "banana_BQ_integrand_2d.png"))
        plt.show()


plt.plot(n_samples_riemann_BQ, integral_vals_riemann_BQ)
plt.savefig(os.path.join(fig_folder, "banana_dist_riemann_convergence.png"))























####################################
# 5) Integrate output from logistic regression model over laplace approximation of posterior  distribution
#    That is: get mean of predictive posterior distribution
####################################

# generate data
num_features = 2
num_classes = 3
train_loader, test_loader = gen_log_regression_data(num_train_samples=100, 
                            num_test_samples=10, 
                            num_features = num_features,
                            num_classes = num_classes,
                            seed = 2,
                            variance=.1,
                            batch_size=0)
X, y = next(iter(train_loader))
plt.scatter(X[:,0], X[:,1], c=y)

# create model
model = LinearModel(num_features=num_features, num_outputs=num_classes, bias=True)

optimizer = torch.optim.Adam(model.parameters(), lr=torch.tensor(.1))

prior_sigma = 1
model, _, _, _, _ = train(model, train_loader=train_loader, test_loader=test_loader, optimizer=optimizer, epochs=500, 
        prior_sigma=prior_sigma, verbose=True, print_every_epoch=100)
print(f"{dict(model.named_parameters())}")

model.eval()
preds = model(X).softmax(dim=-1).argmax(dim=-1)
plt.scatter(X[:,0], X[:,1], c=y, s=100, alpha=.5)
plt.scatter(X[:,0], X[:,1], c=preds, s=20)

accuracy = (preds == y).sum().item()/len(y)
print(f"Accuracy: {accuracy}")

# create Laplace object
laplace = Laplace(model, dataloader=train_loader, prior_sigma=prior_sigma)

# fit Laplace object
mean1, covariance1 = laplace.fit(fitting_type="hessian")
mean2, covariance2 = laplace.fit(fitting_type="GGN")
# Since the model is linear, the two methods should give the same result
print(torch.isclose(mean1, mean2).all(), torch.isclose(covariance1, covariance2).all())

sampler = laplace
parametersubset = dict(model.named_parameters())
xs, ys = next(iter(train_loader))
xs.shape

def make_functional_fwd_xs(_model, post_func=None):
  def fn(parameters, xs):
    res = functional_call(_model, parameters, xs)
    if post_func is not None:
      res = post_func(res)
    return res
  return fn

def functional_loss_from_params(model_func, loss_func, xs, ys):
  # Returns a function that takes parameters, data, and target and returns the loss
  def fn(parameters):
    pred = model_func(parameters, xs)
    loss = loss_func(pred, ys)
    return loss
  return fn

model_func = make_functional_fwd_xs(model, torch.nn.functional.log_softmax)

#loss_func = functional_loss_from_params(model_func, torch.nn.CrossEntropyLoss(), xs, ys)
_=sampler.make_posterior_sample(n_samples=1000)
integral_la, function_values_lp, weights_lp, posterior_samples_la = integrator(sampler, model_func, parametersubset, xs)

pred_class = integral_la.argmax(dim=-1)
(ys == pred_class).sum().item()/len(ys)

