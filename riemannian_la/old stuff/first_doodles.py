import torch
import numpy as np
import matplotlib.pyplot as plt
import math
from torch.utils.data import DataLoader, TensorDataset
from torch_to_pyro import pyro_model_from_pytorch

def fn(xs):
    return torch.column_stack([xs[:,0], xs[:,1], xs[:,0]**2 + xs[:,1]**2])

def generate_data(xs, fn, weights, sigma_y=0.1):
    ys = fn(xs) @ weights + torch.normal(torch.tensor(0.0), torch.tensor(1.)*sigma_y, size=(xs.shape[0],1))
    return ys

class MyModel(torch.nn.Module):
    def __init__(self, M, N, fn):
        super(MyModel, self).__init__()
        self.M = M
        self.N = N
        self.weights = torch.nn.Parameter(torch.rand((M+1,1)))
        self.fn = fn
        
    def forward(self, xs):
        return fn(xs) @ self.weights
    

N=100
M=2
xs = torch.rand((N, M))*1
xs, _ = xs.sort(dim=0)

fn(xs).shape
weights = torch.tensor([[1.0], [1.0], [1.0]])   
ys = generate_data(xs, fn, weights, sigma_y=0.1)
ys.shape

x_test = xs[0:N//5, :]
y_test = ys[0:N//5, :]
x_train = xs[N//5:, :]
y_train = ys[N//5:, :]
x_train.shape, y_train.shape, x_test.shape, y_test.shape

batch_size = 16

train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size)
test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size)



model = MyModel(M, N, fn)
epochs = 10
test_losses = []
epoch_train_losses = []
all_train_losses = []
lrs = []

device="mps"
model.to(device)
optimizer = torch.optim.SGD(params=model.parameters(), lr=torch.tensor(.1))

loss_fn = torch.nn.MSELoss()

scheduler=None

for epoch in range(epochs):
    train_loss = 0
    model.train()
    optimizer.zero_grad()
    num_batches = len(train_loader)
    total_obs = 0
    for i, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        total_obs += len(x)
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        train_loss += loss.item() * len(x)
        if i < len(train_loader)-1:
            all_train_losses.append(loss.item())
            lrs.append(optimizer.param_groups[0]['lr'].item())
        optimizer.step()
        if scheduler:
            scheduler.step()
        print(f"epoch = {epoch}, \tbatch= {i}/{num_batches-1}, train loss: {loss.item():2.5f}, lr: {optimizer.param_groups[0]['lr']:4e}", end="\r")
        
    epoch_train_losses.append(train_loss/ total_obs)

    model.eval()
    test_loss = 0
    
    current_correct_num = 0
    total_obs = 0
    for i, (test_x, test_y) in enumerate(test_loader):
        test_x = test_x.to(device)
        test_y = test_y.to(device)
        total_obs += len(test_x)
        test_pred = model(test_x)
        loss = loss_fn(test_pred, test_y)
        test_loss += loss.item() * len(test_x)
    test_losses.append(test_loss/total_obs)
    
    print(f"epoch = {epoch} \ttrain loss: {epoch_train_losses[-1]:2.5f}, test loss: {test_losses[-1]:2.5f}",
            f", lr: {optimizer.param_groups[0]['lr']:4e}")
    if optimizer.param_groups[0]['lr']<1e-10:
        break

root="/Users/simondanieleiriksson/Documents/DTU-kurser-lokalt/Specialkursus Georgios/riemannian-la"
torch.save(model, f"{root}/models/model.pth")

