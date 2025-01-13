# test if laplace torch works correctly
import laplace, torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset


def KL_divergence(covariance_0, covariance_1):
    covariance_0, covariance_1 = covariance_0.cpu(), covariance_1.cpu()
    logdet0 = torch.logdet(covariance_0)
    logdet1 = torch.logdet(covariance_1)
    return 0.5*(torch.trace(covariance_1.inverse() @ covariance_0) - len(covariance_0) + logdet1 - logdet0)

def mysample(lp, n_samples):
    num_params = len(lp.mean)
    eps = torch.randn(n_samples, num_params)
    posterior_samples = lp.mean + eps @ lp.posterior_scale.T 
    return posterior_samples

class SimpleModel(torch.nn.Module):
    def __init__(self, feature_dim=2):
        super(SimpleModel, self).__init__()
        self.feature_dim = feature_dim
        self.lin1 = torch.nn.Linear(feature_dim, 1, bias=False)
        torch.nn.init.constant_(self.lin1.weight, 1.0)

    def forward(self, x):
        return self.lin1(x)
    
def sum_loss():
    def fn(preds, targets):
        return torch.sum(preds)
    return fn

X = torch.tensor([[1.0, 2.0]])
Y = torch.tensor([3.0])
loader = DataLoader(TensorDataset(X, Y), batch_size=10000)

loss = torch.nn.MSELoss()
model = SimpleModel(feature_dim=X.shape[1])
preds = model(X)

lp = laplace.FullLaplace(model, likelihood="regression", 
                         prior_precision=1.0, sigma_noise=1.0)
lp.fit(loader)

sample = lp.sample(10000000)
empirical_covariance = (sample - sample.mean(dim=0)).T.cov()

print(f"{lp.posterior_covariance = }")
print(f"{empirical_covariance = }")
print(f"{KL_divergence(lp.posterior_covariance, empirical_covariance) = }")

my_sample = mysample(lp, 10000000)
my_covariance = (my_sample - my_sample.mean(dim=0)).T.cov() 
print(f"{my_covariance = }")
print(f"{KL_divergence(lp.posterior_covariance, my_covariance) = }")



def my_other_sample(lp, n_samples):
    dist = torch.distributions.MultivariateNormal(loc=lp.mean, scale_tril=lp.posterior_scale)
    posterior_samples = dist.sample((n_samples,))
    return posterior_samples

my_other_sample = my_other_sample(lp, 10000000)
my_other_covariance = (my_other_sample - my_other_sample.mean(dim=0)).T.cov()
print(f"{my_other_covariance = }")
print(f"{KL_divergence(lp.posterior_covariance, my_other_covariance) = }")

# There appears to be an error in the laplace code. They have forgotton to tranpsose the posterior scale.
