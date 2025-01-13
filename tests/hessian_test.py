import torch
from riemannian_la.hessian import hessian_from_func, hessian_from_model_loss_and_data, hessian_dict_to_matrix

# run a test of hessian_from_func
A = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
B = torch.tensor([[1.0, 1.0]])
C = torch.tensor([1.0])

def second_degree(x):
    return x.T @ A @ x + B @ x + C

x = torch.tensor([1.0, 1.0])
print(f"{second_degree(x) = }")
H = hessian_from_func(second_degree, x)
print(f"{H = }")
print(f"{A + A.T = }")  
assert torch.allclose(H, A + A.T)

# run a test of hessian_from_model_loss_and_data
# we make a very simple model 
class SimpleModel(torch.nn.Module):
    def __init__(self, feature_dim=2):
        super(SimpleModel, self).__init__()
        self.feature_dim = feature_dim
        self.lin1 = torch.nn.Linear(feature_dim, feature_dim, bias=False)
        self.lin2 = torch.nn.Linear(feature_dim, feature_dim, bias=False)
        torch.nn.init.constant_(self.lin1.weight, 2.0)
        torch.nn.init.constant_(self.lin2.weight, 3.0)

    def forward(self, x):
        return self.lin2(self.lin1(x)).sum(dim=1) 

def sum_loss():
    def fn(preds, targets):
        return torch.sum(preds)
    return fn

loss = sum_loss()
xs = torch.tensor([[5.0]])
ys = torch.tensor([1.0])

model = SimpleModel(feature_dim=1)
preds = model(xs)
print(f"{preds = }")
loss(preds, ys)
hess_dict = hessian_from_model_loss_and_data(model, loss, xs, ys)
print(f"{hess_dict = }")

H, params = hessian_dict_to_matrix(hess_dict)
print(f"{H = }") # should return a 2x2 matrix, With 5. in the counterdiagonal and zero in the diagonal
# tensor([[0., 5.], [5., 0.]])

for param in params.keys():
    print(f"{param = }")
    print(f"{params[param] = }")
