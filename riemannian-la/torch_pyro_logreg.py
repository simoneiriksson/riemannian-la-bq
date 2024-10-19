import pyro
import torch
import pyro.distributions as dist
import numpy as np
from torch import nn
from pyro.infer.mcmc import HMC, MCMC, NUTS
from matplotlib import pyplot as plt
import seaborn as sns
from plotting import distributions_plot
from models import MyModel, mini, PyroModel
from pandas.plotting import scatter_matrix
import pandas as pd
from pyro.infer import Predictive

def fn(xs):
    return torch.column_stack([xs[:,0], xs[:,1], xs[:,0]**2 + xs[:,1]**2])

def fn(xs):
    return torch.column_stack([xs, torch.ones_like(xs[:,0])])


def generate_data(xs, fn, weights, sigma_y=0.1):
    ys = fn(xs) @ weights + torch.normal(torch.tensor(0.0), torch.tensor(1.)*sigma_y, size=(xs.shape[0],1))
    return ys

def generate_data(N, mu=None, Sigma=None, class_weights=None, sigma_y=0.1):

    if (mu is None) or (Sigma is None) or (class_weights is None):
        ndim = 2

        class_weights=torch.ones(ndim)*1.0/ndim
        #A = torch.row_stack([torch.eye(ndim).unsqueeze(0), (torch.ones(ndim)- torch.eye(ndim)).unsqueeze(0)])
        Sigma = torch.tensor([[[1.0, 0.0], [0.0, 1.0]], [[1.0, -1.0], [1.0, 1.0]]])*0.2
        mu = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    ndim = mu.shape[1]
    eye = torch.eye(ndim)
    null = torch.zeros(ndim)

    xs_std = torch.distributions.MultivariateNormal(null,eye).sample((100,))
    classes = torch.distributions.Categorical(class_weights).sample((100,))
    xs = torch.bmm(Sigma[classes], xs_std.unsqueeze(-1)).squeeze(-1) + mu[classes] 
    return xs, classes.unsqueeze(1)

M=2
N=100
xs, ys = generate_data(N=N, mu=None, Sigma=None, class_weights=None, sigma_y=0.1)
#xs, ys = generate_data(N=N, mu=torch.tensor([[0.0], [1.0]]), Sigma=torch.tensor([[[0.1]], [[0.1]]]), class_weights=torch.ones(2)/2, sigma_y=0.1)


x_test = xs[0:N//5, :]
y_test = ys[0:N//5]
x_train = xs[N//5:, :]
y_train = ys[N//5:]
x_train.shape, y_train.shape, x_test.shape, y_test.shape

num_classes=1

prior_log_sigma=torch.tensor(.1)
model = MyModel(input_dim=M, num_classes=num_classes, fn=fn, prior_log_sigma=prior_log_sigma)

#model = mini(M_inputs=M, num_hid=10, num_out=1)
log_scale = torch.tensor(.1)
if num_classes == 1:
    likelihood_given_outputs=lambda x: dist.Bernoulli(x.sigmoid())
else:
    likelihood_given_outputs=lambda x: dist.Categorical(x.softmax(-1))

pyromodel = PyroModel(model, prior_log_sigma=prior_log_sigma,
                      likelihood_given_outputs=likelihood_given_outputs,
                      batch_size = None)

nuts_kernel_a = NUTS(pyromodel.model, step_size=1.)
mcmc_auto = MCMC(nuts_kernel_a, num_samples=10000, warmup_steps=500)
mcmc_auto.run(x_train, y_train.type(torch.float32))
mcmc_auto.summary()


def evals(mcmc):
    samples = mcmc.get_samples()
    for param in samples.keys():
        print(f"param = {param}")
        print(f"posterior means = {samples[param].mean(dim=0)}")
        print(f"posterior std = {samples[param].std(dim=0)}")
        print("\n") 

    df = pd.DataFrame(samples[param])
    scatter_matrix(df, alpha = 0.2, figsize = (6, 6), diagonal = 'kde')

print("Auto")
evals(mcmc_auto)

samples = mcmc_auto.get_samples()['parameters_samples']
samples.shape
def decision_boundary(x):
    return -(samples[:,2] + samples[:,0].unsqueeze(0)*x.unsqueeze(1)) / samples[:,1]

xlim=(-1, 3)
x = torch.linspace(xlim[0], xlim[1], 100)
decision_lines = decision_boundary(x)

fig, ax = plt.subplots()
_ = ax.plot(x, decision_lines, color='black', alpha=0.01)
#plot x_train, but collored by the predicted class:
ax.scatter(x_train[:,0], x_train[:,1], c=(y_train.squeeze() == 1).type(torch.float32))
ax.set_xlim(xlim)
ax.set_ylim(xlim)
fig.show()

def decision_boundary_params(samples):
    return torch.column_stack([-samples[:,2]/samples[:,1], -samples[:,0]/samples[:,1]])

dec = decision_boundary_params(samples)
df = pd.DataFrame(dec)
scatter_matrix(df, alpha = 0.2, figsize = (6, 6), diagonal = 'kde')





pred = Predictive(pyromodel.model, mcmc_auto.get_samples())(x_train)
pred_class = pred['pred'].mean(dim=0).sigmoid()>0.5
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_train, pred_class)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

# scatter plot xtrain, but collored by the predicted class:
plt.scatter(x_train[:,0], x_train[:,1], c=pred_class)
plt.scatter(x_train[:,1], y_train)
plt.scatter(x_train[:,0], y_train)
plt.scatter(x_train[:,0], meh["y_pred"].mean(dim=0))


meh2 = Predictive(pyromodel.model, mcmc_auto.get_samples())(x_train)
meh.keys()