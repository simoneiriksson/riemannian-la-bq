# Bayesian Quadrature with Subspace methods for Riemannian Laplace approximation

This repo contains the code for a project I did in spring 2025.

The project presents a method for uncertainty estimation in machine learning models, exploiting the curvature of the loss function, to make efficient parameter samples from the posterior distribution and Bayesian Quadrature to make efficient predictive posterior samples.

Note that the repo has not been cleaned up properly yet - it is in progress.

## Approach
First we present the Vanilla Laplace approximation that approximates the posterior distribution by a Gaussian distribution with the same mean and covariance. This approximation does not work well for high dimensional models, since the posterior distribution tends to be more non-gaussian with increasing dimensionality. 

The idea with the Riemannian Laplace approximation (Riemannian LA) is to exploit the knowledge we have of the loss surface. By introducing a Riemannian metric on the parameter space, we can "guide" the Laplace samples away from areas with low likelihood. However, the computation of Riemannian Laplace samples is highly computationally expensive. 

This leads the contribution of this paper: The Riemannian LA induces a distribution over the parameter space that we want to sample from, in order to get samples from the *predictive* posterior. Assuming that the predictive posterior distribution is drawn from a Gaussian process, we can use samples from the Riemannian LA to retrieve data points as conditions to the Gaussian process. Having a conditional Gaussian process allows us to use Bayesian quadrature to estimate the mean of, and sample the predictive posterior distribution.

Furthermore, by sampling only from a subspace of the full parameter space, the Gaussian process can be restricted to a corresponding subspace. This allows us to integrate the predictive posterior Gaussian process using only a limited number of samples from the Riemannian Laplace approximation.

See the pdf `riemannian_report.pdf` for details.

## How to run
