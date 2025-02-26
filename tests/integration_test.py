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

n_mesh = 100
curvature = 0.0
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
        return self.params.sum().repeat(x.shape[0], 1)**2
        #return self.params.sum().repeat(x.shape[0], 1)+2
        
evaluation_model = tiny_ridiculess_model_class(n_params=2)
functional_evaluation_model = make_functional_fwd_xs(evaluation_model)  # make functional version of model

print(f"{banana_function(xs[0]) = }")

####################################
# 1) discrete integration over function
####################################
#Let's do discrete integration with the banana function
#banana_function = functional_banana(curvature=2.0, sigma_x=2.0, sigma_y=1.0)
#neg_banana = lambda x: -functional_banana(curvature=0.0, sigma_x=1.0, sigma_y=1.0)(x)
span = 10
limits = [[-span, span], [-span, span]]
discrete_sampler = discrete_function_sampler(func=banana_function, limits=limits, n_mesh=n_mesh, 
                                             normalize_weights=False)
posterior_samples, weights = discrete_sampler.samples_and_weights()

banana_function(posterior_samples[0])
sampler = discrete_sampler

parametersubset = dict(evaluation_model.named_parameters())

integral, function_values, weights, posterior_samples = integrator(sampler, functional_evaluation_model, parametersubset, xs)
print(f"1) When using discrete integration over banana FUNCTION we get {integral = }")

plt.scatter(posterior_samples[:,0], posterior_samples[:,1], c=weights)
plt.colorbar()
plt.show()
plt.scatter(posterior_samples[:,0], posterior_samples[:,1], c=function_values)
plt.show()


####################################
# 2) discrete integration over model
####################################
banana_model = Model_from_func(banana_function, input_shape=[2])
parametersubset = dict(banana_model.named_parameters())
x = torch.tensor([0.0, 0.0])
print(f"{banana_function(x) = }")
torch.nn.utils.vector_to_parameters(torch.tensor([0.0, 0.0]), banana_model.parameters())
print(f"{banana_model(x) = }")

xs = torch.tensor([0.0, 0.0]).unsqueeze(0)
ys = banana_function(xs[0]).unsqueeze(0)

discrete_sampler = discrete_model_sampler(banana_model, loss_fn=neglog_loss(), xs=xs, ys=ys, limits=[[-span, span], [-span, span]], n_mesh=n_mesh, normalize_weights=False, prior_sigma=0.0)
posterior_samples, weights = discrete_sampler.samples_and_weights()
integral, function_values, weights, posterior_samples = integrator(sampler, functional_evaluation_model, parametersubset, xs)
print(f"When using discrete integration over bnana MODEL we get {integral = }")
plt.scatter(posterior_samples[:,0], posterior_samples[:,1], c=weights)
plt.colorbar()
plt.show()


####################################
# 3) laplace integration over model
####################################

# and now the laplace approximation with the banana function
banana_model = Model_from_func(banana_function, input_shape=[2])
torch.nn.utils.vector_to_parameters(torch.tensor([0.0, 0.0]), banana_model.parameters())
dict(banana_model.named_parameters())

laplace = Laplace(banana_model, xs=xs, ys=ys, prior_sigma=0, loss_fn=neglog_loss())
laplace.fit(fitting_type="hessian", xs=xs, ys=ys)

for i in range(1, 16):
    n_samples = 2**i
    laplace.make_posterior_sample(n_samples=n_samples)
    integral_la, function_values_lp, weights_lp, posterior_samples_la = integrator(laplace, functional_evaluation_model, parametersubset, xs)
    posterior_samples_la.shape
    print(f"When using the laplace approxiation of posterior of banana MODEL with {n_samples} we get {integral_la = }")

laplace.make_posterior_sample(n_samples=10000)
integral_la, function_values_lp, weights_lp, posterior_samples_la = integrator(laplace, functional_evaluation_model, parametersubset, xs)
laplace.posterior_samples.T.cov()

plt.scatter(posterior_samples_la[:,0].detach(), posterior_samples_la[:,1].detach(), c=weights_lp.detach())
plt.colorbar()
plt.show()

df = pd.DataFrame(posterior_samples_la.detach().numpy())
sns.displot(df, x=0, y=1, kind="kde")
plt.scatter(posterior_samples_la[:,0].detach(), posterior_samples_la[:,1].detach(), c=function_values_lp.detach())
plt.show()


#######################################
# 4) MCMC-integration from banana model
#######################################


# and now the laplace approximation with the banana function
banana_model = Model_from_func(banana_function, input_shape=[2])
torch.nn.utils.vector_to_parameters(torch.tensor([0.0, 0.0]), banana_model.parameters())
const_prior = lambda x: torch.tensor(0.0)
loss_fn = lambda preds, target: torch.sum(preds.log())

parametersubset = dict(banana_model.named_parameters())
sampler = MCMC_sampler(banana_model, parametersubset, xs=xs, ys=ys, loss_fn=neglog_loss(), prior_loss=const_prior)
_=sampler.make_posterior_sample(10000)
sampler.posterior_samples.T.cov()

integral_mcmc, function_values_mcmc, weights_mcmc, posterior_samples_mcmc = integrator(sampler, functional_evaluation_model, parametersubset, xs)

print(f"When using the MCMC sampling of posterior of banana MODEL with {10000} we get {integral_mcmc = }")

plt.scatter(posterior_samples_mcmc[:,0].detach(), posterior_samples_mcmc[:,1].detach(), c=function_values_mcmc.detach())
plt.colorbar()
plt.show()

df = pd.DataFrame(posterior_samples_mcmc.detach().numpy())
sns.displot(df, x=0, y=1, kind="kde")


####################################
# 5) Riemannian laplace integration over model
####################################

# and now the laplace approximation with the banana function
banana_model = Model_from_func(banana_function, input_shape=[2])
torch.nn.utils.vector_to_parameters(torch.tensor([0.0, 0.0]), banana_model.parameters())
parametersubset = dict(banana_model.named_parameters())

R_sampler = Riemann_sampler(banana_model, xs=xs, ys=ys, loss_fn=neglog_loss(), prior_sigma=0)
R_sampler.fit(fitting_type="hessian")

_=R_sampler.make_posterior_sample_la(10000)
R_params = R_sampler.make_posterior_sample()
integral, function_values, weights, posterior_samples = integrator(R_sampler, functional_evaluation_model, parametersubset, xs)
print(f"{integral = }")

fig, ax = riemann_plotter(R_sampler, sample_markers=".", plot_traject=True, plot_traj_marker=None, max_samples=None, LA_arrows=[1])
fig, ax = riemann_plotter(R_sampler, sample_markers=None, plot_traject=False, plot_traj_marker=None)

R_params.T.cov()
R_sampler.posterior_samples_la.T.cov()

for i in range(5, 8):
    n_samples = 2**i
    _=R_sampler.make_posterior_sample_la(n_samples)
    R_params = R_sampler.make_posterior_sample()
    integral_la, function_values_lp, weights_lp, posterior_samples_la = integrator(R_sampler, functional_evaluation_model, parametersubset, xs)
    print(f"When using the Riemannian laplace approxiation of posterior of banana MODEL with {n_samples} we get {integral_la = }")
    ax, fig = riemann_plotter(R_sampler, sample_markers=".", plot_traject=False, plot_traj_marker=None, max_samples=None, LA_arrows=[1])
    plt.show()

















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

