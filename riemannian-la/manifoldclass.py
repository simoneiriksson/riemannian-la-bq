from scipy.integrate import solve_ivp

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from laplace_approx import LA_approximation
from laplace import Laplace
from torchdiffeq import odeint



class Manifold():
    def __init__(self, model, task_type, device="cpu", prior_sigma=0.1, target_sigma=0.1):
        super(Manifold, self).__init__()
        self.model = model.to(device)
        self.task_type = task_type
        self.device = device

        if self.task_type == "regression":
            self.criterion = torch.nn.MSELoss()
        elif self.task_type == "classification":
            self.criterion = torch.nn.CrossEntropyLoss()
        else:
            raise ValueError("Likelihood not recognized")
        self.prior_sigma = prior_sigma if type(prior_sigma) == torch.Tensor else torch.tensor(prior_sigma)
        self.target_sigma = target_sigma if type(target_sigma) == torch.Tensor else torch.tensor(target_sigma)

        self.MAP = None
        self.MAP_covariance = None
        self.LA = Laplace(model, likelihood=self.task_type, hessian_structure="full", subset_of_weights='all')
        
    def set_train_data(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        self.train_loader = DataLoader(TensorDataset(self.x_train, self.y_train), batch_size=16)

    def fit(self, epochs=1, lr=0.1, optimizer=None, verbose=False, print_every_epoch=100):
        if optimizer == None:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            accum_loss = 0            
            for batch_no, (x, y) in enumerate(self.train_loader):
                optimizer.zero_grad()
                x, y = x.to(self.device), y.to(self.device)
                pred = self.model(x)
                loss = self.loss(x, y)
                loss.backward()
                optimizer.step()
                accum_loss += loss.item()
            if verbose:
                if epoch % print_every_epoch == 0:
                    print(f"{epoch = }, {accum_loss = }\t\t ", end="\r")
        self.register_this_as_map()

    def register_this_as_map(self, MAP=None):
        if MAP is None:
            self.MAP = torch.nn.utils.parameters_to_vector(model.parameters()).clone().detach()
            self.LA.fit(self.train_loader)
            self.MAP_covariance = self.LA.posterior_covariance.clone().detach()
        else:
            self.MAP = MAP.clone().detach()
            self.LA.fit(self.train_loader)
            self.MAP_covariance = self.LA.posterior_covariance.clone().detach()
            
    def set_to_map(self):
        if self.MAP is not None:
            torch.nn.utils.vector_to_parameters(self.MAP, self.model.parameters())
        else: raise ValueError("No MAP found")

    def set_weights(self, theta):
        torch.nn.utils.vector_to_parameters(theta, self.model.parameters())

    def loss_bck(self):
        return self.criterion(self.model(self.x_train), self.y_train)
    
    def mse_loss(self, pred=None, target=None):
        if target is None:
            target = self.y_train
        #print(f"{pred.shape = }")
        #print(f"{target.shape = }")
        return (1.0 / self.noise) * self.criterion(pred, target)

    def loss(self, pred=None, target=None):
        #print(f"{x.shape = }")
        #print(f"{y.shape = }")
        if target is None:
            target = self.y_train
        param_norm = torch.nn.utils.parameters_to_vector(self.model.parameters()).norm()**2 / (2 * self.prior_sigma**2)
        #print(f"{param_norm = }")
        #print(f"{pred.shape = }")
        #print(f"{target.shape = }")
        #print(f"{N = }")
        #print(f"{self.target_sigma = }")

        loss_from_pred = self.criterion(pred, target)*x_train.shape[0]/pred.shape[0] / (2 * self.target_sigma**2)
        loss = loss_from_pred + param_norm
        return loss #self.mse_loss(x, y) + self.regularization * torch.nn.utils.parameters_to_vector(model.parameters()).norm()**2

    def gradient_bck(self):
        #params = torch.nn.utils.parameters_to_vector(self.model.parameters())
        grads = torch.autograd.grad(self.loss(), self.model.parameters())
        return torch.cat([parm.flatten() for parm in list(grads)])   
    
    def gradient(self):
        #params = torch.nn.utils.parameters_to_vector(self.model.parameters())
        pred = self.model(self.x_train)
        grads = torch.autograd.grad(self.loss(pred, self.y_train), self.model.parameters())  # D times N
        grads_flat = torch.cat([parm.flatten() for parm in list(grads)])
        return grads_flat

    def covariance(self):
        self.LA.fit(self.train_loader)
        return self.LA.posterior_covariance
    
    def hess(self):
        self.LA.fit(self.train_loader)
        return la.posterior_precision

    def posterior_sample(self, n_samples=1):
        samples = torch.distributions.MultivariateNormal(self.MAP, self.MAP_covariance).sample((n_samples,))
        return samples
        #return self.LA.sample(n_samples=n_samples)

    def velocity_sample(self, n_samples=1):
        samples = torch.distributions.MultivariateNormal(torch.zeros_like(self.MAP), self.MAP_covariance).sample((n_samples,))
        return samples
        #return self.LA.sample(n_samples=n_samples)


    def predictive_samples(self, x, n_samples=1):
        return self.LA.predictive_samples(x, n_samples=n_samples)

    # Analytic derivation of the ODE
    def ode_fun(self, t, state):
        #print(f"{state = }")
        state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
        theta = state_tensor[:self.MAP.shape[0]]
        v = state_tensor[self.MAP.shape[0]:]
        self.set_weights(theta)
        grad_val = self.gradient()
        hess_val = self.hess()
        #print(f"{grad_val = }")
        #print(f"{hess_val = }")
        #print(f"{v = }")

        acc = -(grad_val * (1 / (1 + grad_val.norm()**2)) * (v.T @ hess_val @ v)).flatten()
        return torch.cat([v, acc]).to("cpu").detach().numpy()
    
    def expmap(self, theta, v, rtol=1e-3, atol=1e-3):
        init = torch.cat([theta, v]).to("cpu").detach().numpy()
        solution = solve_ivp(self.ode_fun, [0, 1], init, dense_output=True, rtol=rtol, atol=atol)
        return solution


    # Analytic derivation of the ODE
    def ode_fun_torch_(self, t, state):
        state_tensor = state
        theta = state_tensor[:self.MAP.shape[0]]
        v = state_tensor[self.MAP.shape[0]:]
        self.set_weights(theta)
        grad_val = self.gradient()
        hess_val = self.hess()
        
        acc = -(grad_val * (1 / (1 + grad_val.norm()**2)) * (v.T @ hess_val @ v)).flatten()
        return torch.cat([v, acc])
        
    def expmap_torch(self, theta, v, numsteps=2, rtol=1e-3, atol=1e-3, method="rk4", **kwargs):
        init = torch.cat([theta, v])
        ts = torch.linspace(0, 1, numsteps)
        solution = odeint(self.ode_fun_torch_, init, ts, rtol=rtol, atol=atol, method=method)
        return solution

    def metric(self):
        D = self.MAP.shape[0]
        Id = torch.eye(D)
        Grad_val = self.gradient()
        #Hess_val = self.hess()
        M = Id + Grad_val * Grad_val.T
        return M