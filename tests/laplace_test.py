from riemannian_la.models import LinearModel
from riemannian_la.getdata import gen_log_regression_data
from riemannian_la.train import train
from matplotlib import pyplot as plt

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
model, _, _, _, _ = train(model, train_loader=train_loader, test_loader=test_loader, optimizer=optimizer, epochs=5000, 
        prior_sigma=prior_sigma, verbose=True, print_every_epoch=1000)
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

# make posterior samples
posterior_samples = laplace.make_posterior_sample(n_samples=1000)

# make predictive posterior samples
x_test, y_test = next(iter(test_loader))
predictive_posterior_samples = laplace.predictive_posterior_samples(x_test)
preds_test = predictive_posterior_samples.softmax(dim=-1).argmax(dim=-1)

