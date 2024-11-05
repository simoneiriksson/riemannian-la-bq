from scipy.integrate import solve_ivp

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from laplace_approx import LA_approximation

class Manifold():
    def __init__(self, model, likelihood, device="cpu", regularization=0.1, noise=0.1):
        super(Manifold, self).__init__()
        self.model = model.to(device)
        self.likelihood = likelihood
        self.device = device

        if self.likelihood == "regression":
            self.criterion = torch.nn.MSELoss()
        elif self.likelihood == "classification":
            self.criterion = torch.nn.CrossEntropyLoss()
        else:
            raise ValueError("Likelihood not recognized")
        self.regularization = regularization if type(regularization) == torch.Tensor else torch.tensor(regularization)
        self.noise = noise if type(noise) == torch.Tensor else torch.tensor(noise)

        self.MAP = None
        self.MAP_covariance = None
        
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
        self.MAP = torch.nn.utils.parameters_to_vector(self.model.parameters())
        self.gradient, self.hessian = LA_approximation(self.model, dataloader=self.train_loader, likelihood=self.likelihood, hessian_structure="full", subset_of_weights='all')

    def set_to_map(self):
        if self.MAP is not None:
            torch.nn.utils.vector_to_parameters(self.MAP, self.model.parameters())
        else: raise ValueError("No MAP found")

    def set_weights(self, theta):
        torch.nn.utils.vector_to_parameters(theta, self.model.parameters())

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

        acc = -(grad_val * (1 / (1 + grad_val.T @ grad_val)) * (v.T @ hess_val @ v)).flatten()
        return torch.cat([v, acc]).to("cpu").detach().numpy()
    
    def expmap(self, theta, v):
        init = torch.cat([theta, v]).to("cpu").detach().numpy()
        solution = solve_ivp(self.ode_fun, [0, 1], init, dense_output=True, rtol=1e-3, atol=1e-3)
        return solution

    def metric(self):
        D = self.MAP.shape[0]
        Id = torch.eye(D)
        Grad_val = self.gradient()
        #Hess_val = self.hess()
        M = Id + Grad_val * Grad_val.T
        return M
