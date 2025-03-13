import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


root=".."
experiment_folder = "evaluations"
base_directory = f"{root}/{experiment_folder}"
file = "experiment_eval_2025-03-12_09-47-11.pkl"
tables_folder = f"{root}/report/tables"
plots_folder = f"{root}/report/figures"

filehandle = open(f"{base_directory}/{file}", "rb")
results_list = pickle.load(filehandle)

results_transformed_list = []
for result in results_list:
    res_df = pd.DataFrame.from_dict(result['Results'], dtype="str", orient="index").T.reset_index()
    meta_df = pd.DataFrame.from_dict(result, dtype="str", orient="index").T.drop(columns="Results").reset_index()
    results_transformed_list.append(pd.merge(left=meta_df, right=res_df))

all_res_eval_df = pd.concat(results_transformed_list)
all_res_eval_df["subspace_rank"].fillna(0, downcast='infer')

for integer_col in ["samples", "subspace_rank"]:
    all_res_eval_df[integer_col] = all_res_eval_df[integer_col].fillna(0).astype(int)
for float_col in ['LogLikelihood', 'Expected Calibration Error', 'Accuracy', 'AUROC']:
    all_res_eval_df[float_col] = all_res_eval_df[float_col].astype(float)




# We want the reasult from the max number of samples, for each sampling strategy / rank
max_samples = all_res_eval_df.sort_values(by="samples").groupby(by=["name","subspace_rank"] ).last().reset_index()
max_samples['subspace_rank_txt'] = max_samples['subspace_rank'].apply(str)
max_samples.loc[max_samples['subspace_rank_txt'] == "0", 'subspace_rank_txt'] = "Full rank"

max_samples["name"]

metrics_list = ['Cross Entropy', 'LogLikelihood', 'Expected Calibration Error', 'Accuracy', 'AUROC']
metric = 'LogLikelihood'
methods1 = ["BQ-Riemann", "Laplace", "Riemann"]
methods2 = ["MCMC", "Point estimate"]


table1 = max_samples.where(max_samples["name"].isin(methods1)).pivot_table(index='subspace_rank_txt', columns="name", 
                                                                           values=metric, sort=False)
table1.to_latex(f"{tables_folder}/{metric}_methods1.tex", caption=metric, position='H', float_format="%.4f")  
table2 = max_samples.where(max_samples["name"].isin(methods2)).pivot_table(index='subspace_rank_txt', columns="name", values=metric, sort=False)
table2.to_latex(f"{tables_folder}/{metric}_methods2.tex", caption=metric, position='H', float_format="%.4f")  


# we want a plot of the BQ-Riemann evaluation scores, one graph for each rank, and the number of samples on the x-axis

methods3 = ["BQ-Riemann", "Riemann"]
for method in methods3:
    subset = all_res_eval_df[(all_res_eval_df["name"]==method)]
    if method == "BQ-Riemann": subset.loc[:, "samples"] = (subset["samples"]-1)/6
    table = subset.pivot_table(index='samples', columns="subspace_rank", values=metric, sort=False)
    xticks = range(len(subset['samples'].unique()))
    fig, ax = plt.subplots()
    table.plot(marker="o", colormap="tab10", ax=ax)
    ax.set_title(method)
    ax.set_ylabel(metric)
    ax.tick_params(axis='x', rotation=90)
    fig.savefig(f"{plots_folder}/{method}_{metric}_samples.png")
    ax.set_ylim(-1.2, -.0)

# 