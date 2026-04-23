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

## Implementation


<!--- 
Results:
- Table/plot with the headline outcome.
- Link to results/ artifacts.
--->
## How to run


Create a virtual environment and install the packages:
```
python -m venv .venv 
source .venv/bin/activate
pip install -e .
```
Then the central experiment in the report can be run with 
```
cd riemannian_la/examples
python3 classification_test_experiment.py
```
A few notebooks are also available: `report_integration_test.ipynb` and `report.ipynb`.

## Repo layout
<!--- 
Repo layout
- What’s in src/, evaluation/, etc.
--->
````
.
├── setup.py
├── requirements.txt
├── readme.md
├── riemannian_report.pdf
│
├── riemannian_la/                  # Core package
│   ├── __init__.py
│   ├── BQ_rays_subspace.py
│   ├── FullGaussianMeasure.py
│   ├── GGN_hessian.py
│   ├── MCMC_sampler.py
│   ├── RayAcquisition.py
│   ├── classification_eval.py
│   ├── discrete_sampler.py
│   ├── getdata.py
│   ├── hessian.py
│   ├── integration.py
│   ├── laplace_approx.py
│   ├── models.py
│   ├── riemann_sampler.py
│   ├── train.py
│   └── utils.py
│
└── examples/                       # Example scripts and notebooks
    ├── UCI_test.py
    ├── classification_test_experiment.py
    ├── emukit_doodles.py
    ├── evalplots.py
    ├── report.ipynb
    └── report_integration_test.ipynb
````

<!----
│
├── evaluations/                    # Experiment evaluation outputs
│   ├── experiment_eval_*.pkl
│   ├── experiment_eval_*.txt
│   ├── synth_experiment_eval_*.pkl
│   └── synth_experiment_eval_*.txt
│
├── figures/                        # Experiment figures (multiple runs)
│   ├── classification_experiment/
│   ├── classification_experiment2/
│   ├── classification_experiment3/
│   ├── classification_experiment4/
│   └── ...
│
├── report/
│   ├── figures/
│   │   └── ...                     # Report figures
│   └── tables/
│       ├── LogLikelihood_methods1.tex
│       ├── LogLikelihood_methods2.tex
│       ├── UCI_wine1.tex
│       ├── UCI_wine2.tex
│       ├── synth1.tex
│       └── synth2.tex
--->


<!--- 
Citation / attribution
- BibTeX or a short citation line if it maps to a paper/report.
--->