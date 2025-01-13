import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Subset
from torch.distributions import Categorical, MultivariateNormal
from matplotlib import pyplot as plt

from contextlib import contextmanager

@contextmanager
def torch_seed(seed):
    """
    A context manager to temporarily set the random seed in PyTorch.
    
    Args:
        seed (int): The seed value to use within the context.
    """
    # Save the current random state
    random_state = torch.get_rng_state()
    try:
        torch.manual_seed(seed)
        yield
    finally:
        # Restore the previous random state
        torch.set_rng_state(random_state)

def make_loaders(X, y, train_size, batch_size=0):
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    subset_test = Subset(test_dataset, indices=range(len(test_dataset) // 1))
    subset_train = Subset(train_dataset, indices=range(len(train_dataset) // 1))
    if batch_size == 0:
        batch_size = max(len(subset_train), len(subset_test))
    train_loader = DataLoader(subset_train, batch_size=batch_size)
    test_loader = DataLoader(subset_test, batch_size=batch_size)
    return train_loader, test_loader

def gen_linear_regression_data(num_train_samples=10, 
                              num_test_samples=10, 
                              noise_std=1.0, 
                              batch_size=0):
    def fn(x):
        return 2 - 1*x 
    N = num_train_samples + num_test_samples
    X = torch.rand(N, 1)*2-1
    y = fn(X) + torch.randn_like(X) * noise_std
    return make_loaders(X, y, num_train_samples, batch_size)    

def gen_log_regression_data(num_train_samples=10, 
                              num_test_samples=10, 
                              num_features = 1,
                              num_classes = 2,
                              variance = .1,
                              batch_size=0):
    num_samples = num_train_samples + num_test_samples
    with torch_seed(2):
        means = torch.randn(num_classes, num_features)
        class_weights = torch.nn.functional.softmax(torch.randn(num_classes))
        covariances = torch.eye(num_features).repeat(num_classes, 1, 1)*variance

        # Create a categorical distribution for class selection
        class_dist = Categorical(probs=class_weights)
        # Generate class labels
        y = class_dist.sample((num_samples,))
        x = torch.zeros((num_samples, num_features))
        for k in range(num_classes):
            class_mask = y == k
            num_class_samples = class_mask.sum()
            if num_class_samples > 0:
                mvn = MultivariateNormal(means[k], covariances[k])
                x[class_mask] = mvn.sample((num_class_samples,))
    return make_loaders(x, y, num_train_samples, batch_size)    

if __name__ == "main":
    # test the data generation functions
    # Linear regression:
    train_loader, test_loader = gen_curve_regression_data(num_train_samples=100)
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


