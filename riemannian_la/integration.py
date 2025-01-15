
import os
import sys
# set working directory
#os.chdir("../riemannian_la")
print(os.getcwd())
from models import LinearModel, Model_from_func
from getdata import gen_log_regression_data
from train import train
from laplace_approx import Laplace, vector_to_parameterdict
from utils import loss_func_from_target_sigma, make_functional_fwd_xs, functional_loss, functional_loss_for_vmap, sum_loss, neglog_loss
from torch.func import grad, jvp, vjp, hessian, jacfwd, jacrev, vmap, functional_call


from matplotlib import pyplot as plt
import torch


class discrete_model_sampler:
    def __init__(self, model, loss_fn, xs, ys, limits, n_mesh=100, normalize_weights=True, parametersubset=None):
        if parametersubset is None:
            self.parametersubset = dict(model.named_parameters())
        else:
            self.parametersubset = parametersubset
        self.model = model
        self.loss_fn = loss_fn
        self.xs = xs
        self.ys = ys
        self.samples = None
        self.weights = None
        self.limits = limits
        self.n_mesh = n_mesh
        self.discrete_sampler = True
        self.span = [abs(l[1] - l[0])  for l in self.limits]
        self.size = torch.tensor(self.span).prod()
        self.dims = len(self.limits)
        self.tile_size = self.size / self.n_mesh**self.dims
        self.normalize_weights = normalize_weights
        self.mesh_vals = [torch.linspace(l[0]+self.span[i]/self.n_mesh, l[1]-self.span[i]/self.n_mesh, self.n_mesh) for i, l in enumerate(self.limits)]

    def samples_and_weights(self):
        # make meshgrid
        meshgrid = torch.meshgrid(self.mesh_vals)
        meshgrid = torch.stack(meshgrid, dim=-1)
        samples = meshgrid.view(-1, len(self.limits))
        # weights here should be equal to the likelihood of the data given the parameters
        model_functional = make_functional_fwd_xs(self.model)  # get me the functional version of the model
        loss_functional = functional_loss_for_vmap(model_functional, self.parametersubset, self.loss_fn, xs, ys)
        weights = vmap(loss_functional)(samples)* self.tile_size
        if self.normalize_weights:
            weights = weights / (weights.sum())  # Does this make sense at all?
        return samples, weights

class discrete_function_sampler:
    def __init__(self, func, limits, n_mesh=100, normalize_weights=False):
        self.func = func
        self.samples = None
        self.weights = None
        self.limits = limits
        self.n_mesh = n_mesh
        self.discrete_sampler = True
        self.span = [abs(l[1] - l[0])  for l in self.limits]
        self.size = torch.tensor(self.span).prod()
        self.dims = len(self.limits)
        self.tile_size = self.size / self.n_mesh**self.dims
        self.normalize_weights = normalize_weights
        self.mesh_vals = [torch.linspace(l[0]+self.span[i]/self.n_mesh, l[1]-self.span[i]/self.n_mesh, self.n_mesh) for i, l in enumerate(self.limits)]

    def samples_and_weights(self):
        # make meshgrid
        meshgrid = torch.meshgrid(self.mesh_vals)
        meshgrid = torch.stack(meshgrid, dim=-1)
        samples = meshgrid.view(-1, len(self.limits))
        weights = vmap(self.func)(samples).view(-1) * self.tile_size
        if self.normalize_weights:
            weights = weights / (weights.sum())  # Does this make sense at all?
        return samples, weights

def integrator(sampler, model_func, parametersubset, xs):
    if hasattr(sampler, "discrete_sampler"):
        posterior_samples, weights = sampler.samples_and_weights()
    else:
        posterior_samples = laplace.posterior_samples
        weights = torch.ones(posterior_samples.shape[0])/posterior_samples.shape[0]
        
    # loop over posterior samples
    for sample_no, posterior_sample in enumerate(posterior_samples):
        param_dict = vector_to_parameterdict(posterior_sample, parametersubset)
        function_value = model_func(param_dict, xs)
        if sample_no == 0: 
            function_values = torch.zeros((len(posterior_samples), *function_value.shape))
        function_values[sample_no] = function_value 

    integral = (function_values * weights[:, None, None] ).sum(dim=0)
    return integral, function_values, weights, posterior_samples





# Let's do discrete integration with the banana function
from models import functional_banana

#banana_function = functional_banana(curvature=2.0, sigma_x=2.0, sigma_y=1.0)
#neg_banana = lambda x: -functional_banana(curvature=0.0, sigma_x=1.0, sigma_y=1.0)(x)
banana_function = functional_banana(curvature=0.0, sigma_x=2.0, sigma_y=1.0)
x = torch.tensor([1.0, 1.0])
print(f"{banana_function(x) = }")
span = 10
discrete_sampler = discrete_function_sampler(func=banana_function, limits=[[-span, span], [-span, span]], n_mesh=3, normalize_weights=False)
posterior_samples, weights = discrete_sampler.samples_and_weights()
weights.sum()
w = weights

banana_function(posterior_samples[0])
sampler = discrete_sampler

class tiny_ridiculess_model_class(torch.nn.Module):
    def __init__(self, n_params):
        super().__init__()
        self.params = torch.nn.Parameter(torch.arange(n_params).float())
    def forward(self, x):
        #return torch.sin(self.params*3.1).sum().repeat(x.shape[0], 1)**2
        #return torch.ones(x.shape[0], 1)
        return self.params.sum().repeat(x.shape[0], 1)**2
        
evaluation_model = tiny_ridiculess_model_class(n_params=2)
evaluation_model(torch.tensor([[1.0, 1.0]]))

functional_evaluation_model = make_functional_fwd_xs(evaluation_model)

# make predictive posterior samples
xs = torch.tensor([0.0, 0.0]).unsqueeze(0)
ys = banana_function(xs[0]).unsqueeze(0)
parametersubset = dict(evaluation_model.named_parameters())

integral, function_values, weights, posterior_samples = integrator(sampler, functional_evaluation_model, parametersubset, xs)
print(f"{integral = }")


plt.scatter(posterior_samples[:,0], posterior_samples[:,1], c=weights)
plt.colorbar()
plt.show()
plt.scatter(posterior_samples[:,0], posterior_samples[:,1], c=function_values)

# and now the laplace approximation with the banana function
banana_model = Model_from_func(banana_function, input_shape=[2])
torch.nn.utils.vector_to_parameters(torch.tensor([0.0, 0.0]), banana_model.parameters())
dict(banana_model.named_parameters())


laplace = Laplace(banana_model, dataloader=None, prior_sigma=0, loss_fn=neglog_loss())
laplace.fit(fitting_type="hessian", xs=xs, ys=ys)

for i in range(1, 6):
    n_samples = 10**i
    laplace.make_posterior_sample(n_samples=n_samples)
    integral_lp, function_values_lp, weights_lp, posterior_samples_lp = integrator(laplace, functional_evaluation_model, parametersubset, xs)
    posterior_samples_lp.shape
    print(f"{n_samples}, {integral_lp = }")

plt.scatter(posterior_samples_lp[:,0].detach(), posterior_samples_lp[:,1].detach(), c=weights_lp.detach())
plt.colorbar()
plt.show()

plt.scatter(posterior_samples_lp[:,0].detach(), posterior_samples_lp[:,1].detach(), c=function_values_lp.detach())



#################
# Now we do the same, but with model-version
#################


banana_function = functional_banana(curvature=0.0, sigma_x=2.0, sigma_y=1.0)
banana_model = Model_from_func(banana_function, input_shape=[2])
parametersubset = dict(banana_model.named_parameters())
x = torch.tensor([0.0, 0.0])
print(f"{banana_function(x) = }")
torch.nn.utils.vector_to_parameters(torch.tensor([0.0, 0.0]), banana_model.parameters())
print(f"{banana_model(x) = }")

xs = torch.tensor([0.0, 0.0]).unsqueeze(0)
ys = banana_function(xs[0]).unsqueeze(0)


span = 10
discrete_sampler = discrete_model_sampler(banana_model, loss_fn=sum_loss(), xs=xs, ys=ys, limits=[[-span, span], [-span, span]], n_mesh=3, normalize_weights=False)
posterior_samples, weights = discrete_sampler.samples_and_weights()
weights.sum()













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

integral_lp, function_values_lp, weights_lp, posterior_samples_lp = integrator(sampler, model_func, parametersubset, xs)

