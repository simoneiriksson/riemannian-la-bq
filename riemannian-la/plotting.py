import torch
from matplotlib import pyplot as plt 
import seaborn as sns

def distributions_plot(posterior_samples_tensor, LA_loc, LA_cov):
    D = posterior_samples_tensor.shape[1]
    # Create subplots for every pair of dimensions
    fig, axs = plt.subplots(D, D, figsize=(15, 15))
    for i in range(D):
        for j in range(D):

            ax = axs[j, i]
            if i == j:
                post = posterior_samples_tensor[:,i]
                sns.histplot(post, kde=True, ax=ax, stat="density")
                xs = torch.linspace(post.min(), post.max(), 100)
                sns.lineplot(x=xs, y=torch.distributions.Normal(LA_loc[i], torch.sqrt(LA_cov[i,i])).log_prob(xs).exp(), ax=ax, color="red")
                ax.set_xlabel(f"Dim {i}")
            #i=0; j=1
            #fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            else:
                xi, xj = posterior_samples_tensor[:, i], posterior_samples_tensor[:, j]

                xij = torch.stack([xi, xj], dim=0)
                mean = xij.mean(dim=1)
                cov = xij.cov()
                # Scatter plot for the pair of dimensions (i, j)
                ax.scatter(xi, xj, s=10, label=f"Dim {i} vs Dim {j}")
                ax.set_xlabel(f"Dim {i}")
                ax.set_ylabel(f"Dim {j}")
                stds = torch.tensor([1, 2, 3, 4])


                # plot the posterior distribution
                eigvals, eigvecs = torch.linalg.eigh(cov)
                angle = torch.rad2deg(torch.atan2(eigvecs[1, 0], eigvecs[0, 0]))
                width, height = torch.sqrt(eigvals).unsqueeze(1) * 2 * stds.unsqueeze(0)
                for curveno in range(stds.shape[0]):
                    ellipse = plt.matplotlib.patches.Ellipse(mean, width[curveno], height[curveno], angle=angle, fill=False, edgecolor="blue", linewidth=2)
                    ax.add_patch(ellipse)

                # plot the laplace approximation
                eigvals, eigvecs = torch.linalg.eigh(LA_cov[[i,j],:][:,[i,j]])
                angle = torch.rad2deg(torch.atan2(eigvecs[1, 0], eigvecs[0, 0]))
                width, height = torch.sqrt(eigvals).unsqueeze(1) * 2 * stds.unsqueeze(0)
                for curveno in range(stds.shape[0]):
                    ellipse = plt.matplotlib.patches.Ellipse(LA_loc[[i,j]], width[curveno], height[curveno], angle=angle, fill=False, edgecolor="red", linewidth=2)
                    ax.add_patch(ellipse)

                #ax.set_xlim(posterior_samples_tensor[:,i].min(), posterior_samples_tensor[:,i].max())
                #ax.set_ylim(posterior_samples_tensor[:,j].min(), posterior_samples_tensor[:,j].max())
    return fig, axs