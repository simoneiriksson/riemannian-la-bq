
###########################################
# Getting Hessian for linearized manifold
from torch.func import grad, jvp, vjp, hessian, jacfwd, jacrev, vmap, functional_call
likelihood_given_outputs=lambda x: CustomNormal(x, target_log_sigma.exp())


def make_functional_fwd(_model):
    def fn(data, parameters):
        return functional_call(_model, parameters, (data,))
    return fn

def make_loss_func_from_distr_per_obs(model_func, prior_distribution, likelihood_given_outputs):
    def fn(parameters, data, target):
        param_vector = torch.nn.utils.parameters_to_vector([param for param in parameters.values()]).detach()
        pred = model_func(data, parameters)
        like = likelihood_given_outputs(pred).log_prob(target)
        print(f"{like = }")
        print(f"{like.shape = }")
        reg = prior_distribution.log_prob(param_vector)
        print(f"{-(like.sum() + reg) = }")
        return -(like.sum() + reg)
    return fn
model = MODEL(num_params=M, fn=FN, prior_log_sigma=prior_log_sigma)
from laplace_approx import LA_approximation
#grad_, hess_ = LA_approximation(model, xs=x_train, ys=y_train[:,0], task_type="regression", prior_sigma=prior_log_sigma.exp(), target_sigma=target_log_sigma.exp())

parametersubset = model.named_parameters
num_params =  sum([p[1].numel() for p in model.named_parameters()])
params_used = dict(parametersubset())  # parameters used in the loss function
#params_used = tuple(parametersubset())  # parameters used in the loss function
#[param[1] for param in params_used]
prior_distribution = torch.distributions.MultivariateNormal(torch.zeros(num_params), torch.eye(num_params) * prior_log_sigma.exp() * num_params)

dataloader = DataLoader(TensorDataset(xs, ys), batch_size=16)
x, y = next(iter(dataloader))
y = y.squeeze(-1)
model_func = make_functional_fwd(model)

loss_func = make_loss_func_from_distr_per_obs(model_func, prior_distribution, likelihood_given_outputs)
ft_comute_grad = grad(loss_func)
meh1 = ft_comute_grad(params_used, x, y)
meh1.keys()
print(f"{meh1['fc1.bias'] = }")

ft_comute_grad = grad(loss_func, argnums=0)
mapped_ft = vmap(ft_comute_grad, in_dims=(None, 0, 0))
res = mapped_ft(params_used, x, y)
print(f"{res['fc1.bias'].shape = }")
