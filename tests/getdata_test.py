#from ..riemannian_la.getdata import gen_linear_regression_data, gen_log_regression_data
from ..getdata import gen_linear_regression_data, gen_log_regression_data

import riemannian-la.getdata

# test the data generation functions
# Linear regression:
train_loader, test_loader = gen_linear_regression_data(num_train_samples=100)
X, y = next(iter(train_loader))
plt.scatter(X, y)

# Logistic regression, 1d:
train_loader, test_loader = gen_log_regression_data(num_train_samples=100, 
                            num_test_samples=10, 
                            num_features = 1,
                            num_classes = 4,
                            batch_size=0)
X, y = next(iter(train_loader))
plt.scatter(X, y, c=y)

# Logistic regression, 2d:
train_loader, test_loader = gen_log_regression_data(num_train_samples=100, 
                            num_test_samples=10, 
                            num_features = 2,
                            num_classes = 3,
                            seed = 2,
                            variance=.1,
                            batch_size=0)
X, y = next(iter(train_loader))
plt.scatter(X[:,0], X[:,1], c=y)


