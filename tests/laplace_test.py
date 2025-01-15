import os
import sys
# set working directory
os.chdir("../riemannian_la")
print(os.getcwd())
from models import LinearModel
from getdata import gen_log_regression_data, gen_linear_regression_data
from train import train
from laplace_approx import Laplace
from utils import tensify
from matplotlib import pyplot as plt
import torch

################################
# Logistic regression example
################################

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
xs, ys = next(iter(train_loader))
plt.scatter(xs[:,0], xs[:,1], c=ys)

# create model
model = LinearModel(num_features=num_features, num_outputs=num_classes, bias=True)

optimizer = torch.optim.Adam(model.parameters(), lr=torch.tensor(.1))

prior_sigma = 1
model, _, _, _, _ = train(model, train_loader=train_loader, test_loader=test_loader, optimizer=optimizer, epochs=5000, 
        prior_sigma=prior_sigma, verbose=True, print_every_epoch=1000)
print(f"{dict(model.named_parameters())}")

model.eval()
preds = model(xs).softmax(dim=-1).argmax(dim=-1)
plt.scatter(xs[:,0], xs[:,1], c=ys, s=100, alpha=.5)
plt.scatter(xs[:,0], xs[:,1], c=preds, s=20)

accuracy = (preds == ys).sum().item()/len(ys)
print(f"Accuracy: {accuracy}")

# create Laplace object
laplace = Laplace(model, dataloader=train_loader, prior_sigma=prior_sigma)

# fit Laplace object
mean1, covariance1 = laplace.fit(fitting_type="hessian")
mean2, covariance2 = laplace.fit(fitting_type="GGN")
# Since the model is linear, the two methods should give the same result
print(torch.isclose(mean1, mean2).all(), torch.isclose(covariance1, covariance2).all())

# make posterior samples
posterior_samples = laplace.make_posterior_sample(n_samples=1000)

# make predictive posterior samples
x_test, y_test = next(iter(test_loader))
predictive_posterior_samples = laplace.predictive_posterior_samples(x_test)
preds_test = predictive_posterior_samples.softmax(dim=-1).argmax(dim=-1)

################################
# Linear regression example
################################
target_sigma =1.
prior_sigma = 100.

train_loader, test_loader = gen_linear_regression_data(num_train_samples=10, 
                              num_test_samples=10, 
                              target_sigma=target_sigma, 
                              batch_size=0, seed=2)
xs, ys = next(iter(train_loader))
#plt.scatter(xs, ys)
#plt.show()

# create model
model = LinearModel(num_features=1, num_outputs=1, bias=True)

optimizer = torch.optim.Adam(model.parameters(), lr=torch.tensor(.1))
model, _, _, _, _ = train(model, train_loader=train_loader, test_loader=test_loader, optimizer=optimizer, epochs=1000, 
        prior_sigma=prior_sigma, verbose=True, print_every_epoch=1000, target_sigma=target_sigma)
print(f"{dict(model.named_parameters())}")

#model.eval()
#preds_MAP = model(xs)

#plt.scatter(xs, ys, s=100, alpha=.5)
#plt.scatter(xs, preds_MAP.detach(), s=20)

# create Laplace object
laplace = Laplace(model, dataloader=train_loader, prior_sigma=prior_sigma, target_sigma=target_sigma)

# fit Laplace object
mean1, precision1 = laplace.fit(fitting_type="hessian")
mean2, precision2 = laplace.fit(fitting_type="GGN")

# Since the model is linear, the two methods should give the same result
print(torch.isclose(mean1, mean2).all(), torch.isclose(precision1, precision2).all())

# make posterior samples
posterior_samples = laplace.make_posterior_sample(n_samples=1000)
xs_plot = torch.linspace(-1, 1, 100).unsqueeze(1)
preds_samples = laplace.predictive_posterior_samples(xs_plot)
preds_mean = preds_samples.mean(dim=0)
preds_std = preds_samples.std(dim=0)
preds_MAP = model(xs_plot)

plt.scatter(xs, ys, s=100, alpha=.5, label="True")
plt.plot(xs_plot, preds_MAP.detach(), label="MAP")
plt.plot(xs_plot, preds_mean.detach(), label="Mean")
plt.plot(xs_plot, preds_mean.detach() + 2*preds_std.detach())
plt.plot(xs_plot, preds_mean.detach() - 2*preds_std.detach())
plt.plot(xs_plot, preds_mean.detach() + 2*preds_std.detach() + 2*torch.sqrt(tensify(target_sigma)**2))
plt.plot(xs_plot, preds_mean.detach() - 2*preds_std.detach() - 2*torch.sqrt(tensify(target_sigma)**2))
plt.legend()


# Analytical solution:
Phi = torch.cat([xs, torch.ones(xs.shape[0],1)], dim=1)
prior_Sigma = torch.eye(2) *prior_sigma**2
prior_mu = torch.zeros(2)

S_inverse = prior_Sigma.inverse() + target_sigma**-2 * Phi.T @ Phi
mu = S_inverse.inverse() @ (prior_Sigma.inverse() @ prior_mu + target_sigma**-2 * (Phi.T @ ys.squeeze()))
print(f"Analytical solution:")
print(f"{mu=}")
print(f"{S_inverse=}")
