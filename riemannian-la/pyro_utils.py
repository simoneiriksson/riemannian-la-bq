from pyro.contrib.autoguide import AutoDiagonalNormal, AutoMultivariateNormal
from pyro.infer import MCMC, NUTS, HMC, SVI, Trace_ELBO
from pyro.optim import Adam
import seaborn as sns
from pyro.infer.autoguide.guides import AutoLaplaceApproximation
from pyro.ops.hessian import hessian
import pyro


##########################################
# Laplace approximation
def get_laplace(pyro_model=None, x=None, y=None, num_iters=400, lr=0.01, device="cpu"):
    delta_guide = AutoLaplaceApproximation(pyro_model)
    svi = SVI(pyro_model, delta_guide, Adam({"lr": lr}), loss=Trace_ELBO())
    pyro.clear_param_store()
    if x is not None:
        for i in range(num_iters):
            loss = svi.step(x.to(device), y.to(device))
            if i % 1000 == 0:
                print(f"i = {i}, loss = {loss}")
        #svi.run(x_train.to(device), y_train[:,0].to(device), num_iters)
        guide_trace = pyro.poutine.trace(delta_guide).get_trace(x)
        model_trace = pyro.poutine.trace(pyro.poutine.replay(delta_guide.model, trace=guide_trace)).get_trace(x)
    else:
        for i in range(num_iters):
            loss = svi.step(y=y)
            if i % 1000 == 0:
                print(f"i = {i}, loss = {loss}")
        guide_trace = pyro.poutine.trace(delta_guide).get_trace()
        model_trace = pyro.poutine.trace(pyro.poutine.replay(delta_guide.model, trace=guide_trace)).get_trace() 
    loss = guide_trace.log_prob_sum() - model_trace.log_prob_sum()
    H = hessian(loss, delta_guide.loc)
    loc = delta_guide.loc.detach()
    cov = H.inverse()
    model_guide_parameters = {}
    for k in model_trace.nodes.keys():
        #print(f"{k=}")
        if model_trace.nodes[k]["type"] == "sample":
            #print(f"{model_trace.nodes[k]['value'].shape=}")
            model_guide_parameters[k] = model_trace.nodes[k]["value"].shape


    return loc, cov, H, delta_guide, model_guide_parameters
