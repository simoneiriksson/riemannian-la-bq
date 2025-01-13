import torch
import math
# kindly provided by ChatGPT
# For some reason there is a .item() in the return statement of the log_prob method in the torch.distributiuon.Normal class
# This does not work with torch.func.vmap 
# So here is a custom Normal distribution class that does not have the .item() in the log_prob method
class CustomNormal:
    def __init__(self, mean, std):
        """
        Initialize the custom Normal distribution.
        
        Args:
            mean (torch.Tensor): The mean of the normal distribution.
            std (torch.Tensor): The standard deviation of the normal distribution.
        """
        self.mean = mean
        self.std = std
        self.variance = std ** 2

    def log_prob(self, value):
        """
        Calculate the log probability of `value` under the normal distribution.
        
        Args:
            value (torch.Tensor): The value(s) for which to compute the log probability.
            
        Returns:
            torch.Tensor: Log probability of the value(s).
        """
        # Log probability formula
        log_prob = -0.5 * ((value - self.mean) ** 2 / self.variance + torch.log(2 * torch.pi * self.variance))
        return log_prob

    def sample(self):
        """
        Generate a sample from the normal distribution.
        
        Returns:
            torch.Tensor: A sample from the normal distribution.
        """
        eps = torch.randn_like(self.mean)
        return self.mean + self.std * eps

    def rsample(self):
        """
        Generate a reparameterized sample from the normal distribution.
        
        Returns:
            torch.Tensor: A reparameterized sample (differentiable) from the normal distribution.
        """
        eps = torch.randn_like(self.mean)
        return self.mean + self.std * eps

    def entropy(self):
        """
        Calculate the entropy of the normal distribution.
        
        Returns:
            torch.Tensor: Entropy of the normal distribution.
        """
        return 0.5 * torch.log(2 * torch.pi * math.e * self.variance)