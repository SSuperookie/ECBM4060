import scanpy as sc
import pandas as pd
import scvelo as scv
import seaborn as sns
import matplotlib.pyplot as plt


count_matrix_path = r'D:\karon\pycharm\ECBM4060_project\GSE166181_Normalized_UMI_CountMatrix.tsv'
metadata_path = r'D:\karon\pycharm\ECBM4060_project\GSE166181_Metadata.tsv'

# laod and preprocess
def load_and_preprocess(count_path, meta_path):
    adata = sc.read_csv(count_path, delimiter='\t')
    adata = adata.transpose()  
    metadata = pd.read_csv(meta_path, delimiter='\t')
    adata.obs = metadata  
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    adata = adata[:, adata.var.highly_variable].copy()

    return adata


adata = load_and_preprocess(count_matrix_path, metadata_path)

def run_dimensionality_reduction_and_clustering(adata):
    # PCA
    #sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, svd_solver="arpack")
    # TIME
    #adata.obs['Condition'] = adata.obs['time']
    # umap
    print(f"n_neighbors=20, n_pcs=50, shape={adata.shape}")
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
    sc.tl.leiden(adata, resolution=0.8, flavor="igraph", n_iterations=2, directed=False)
    # cluster
    sc.tl.umap(adata)

    return adata

adata = run_dimensionality_reduction_and_clustering(adata)

# Create a mapping for the clusters
cluster_names = {
    '6': 'MAIT'
}
# Map the cluster names to a new column 'cell_type' in the .obs attribute
adata.obs['cell_type'] = adata.obs['leiden'].map(cluster_names)
#  cell_type= 'MAIT'
mait_adata = adata[adata.obs['cell_type'] == 'MAIT'].copy()


def refine_subclusters(adata):
    subcluster_counts = adata.obs['leiden'].value_counts()
    valid_subclusters = subcluster_counts[subcluster_counts > 10].index
    adata_valid = adata[adata.obs['leiden'].isin(valid_subclusters)].copy()
    return adata_valid


adata_valid = refine_subclusters(mait_adata)

#  pseudotime_analysis
def pseudotime_analysis(adata):
    # DEA
    sc.tl.rank_genes_groups(adata, 'time', method='wilcoxon', reference='R T0')
    sc.pl.rank_genes_groups(adata, n_genes=10, sharey=False)  

    # Suppose CD74 is a marker gene for T0
    if 'CD74' not in adata.var_names:
        raise ValueError("The specified root gene 'CD74' is not in the dataset.")

    # Set the root node using the specified gene, such as CD74
    cd74_expression = adata[:, 'CD74'].X
    if cd74_expression.ndim == 2:
        cd74_expression = cd74_expression.toarray().flatten()  # If sparse matrix, convert it to an array

    root_cell_idx = cd74_expression.argmin()  # found highest expression CD74 cell
    root_cell_name = adata.obs_names[root_cell_idx]
    adata.uns['iroot'] = root_cell_idx
    print(f"Root cell selected: {root_cell_name} (lowest CD74 expression)")

    sc.tl.diffmap(adata, n_comps=20)  
    sc.tl.dpt(adata)

    
    quantile_threshold = 0.95
    max_threshold = adata.obs['dpt_pseudotime'].quantile(quantile_threshold)
    print(f"Dynamic max threshold (quantile {quantile_threshold}): {max_threshold:.4f}")

    # Screen out cells with too much false time
    valid_cells = adata.obs['dpt_pseudotime'] <= max_threshold
    adata = adata[valid_cells].copy()
    print(f"Filtered cells with pseudotime > {max_threshold}. Remaining cells: {adata.n_obs}")


    # Standardize the pseudo-time
    adata.obs['dpt_pseudotime'] = (adata.obs['dpt_pseudotime'] - adata.obs['dpt_pseudotime'].min()) / \
                                  (adata.obs['dpt_pseudotime'].max() - adata.obs['dpt_pseudotime'].min())

    # visualization
    sc.pl.umap(adata, color='dpt_pseudotime', title=f"Pseudotime of MAIT Cells (Filtered, Root from CD74)")
    return adata

adata_valid = pseudotime_analysis(adata_valid)


#Pseudo-time and RNA velocity were analyzed by time point and response type
def analyze_conditions(adata):
    adata.obs['Response'] = adata.obs['time'].str.split(' ').str[0]


    plt.figure(figsize=(10, 6))
    sns.boxplot(data=adata.obs, x='Response', y='dpt_pseudotime')
    plt.title("Pseudotime Distribution: Responders (R) vs Non-Responders (NR)")
    plt.xlabel("Response")
    plt.ylabel("Pseudotime")
    plt.show()

analyze_conditions(adata_valid)

# pseudotime distribution
def plot_pseudotime_distribution(adata):
    # boxplot and violin
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=adata.obs, x='time', y='dpt_pseudotime', hue='Response', palette="Set2")
    plt.title("Pseudotime Distribution Across Conditions and Responses")
    plt.xlabel("Condition (Time Points)")
    plt.ylabel("Pseudotime")
    plt.legend(title="Response")
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.violinplot(data=adata.obs, x='time', y='dpt_pseudotime', hue='Response', split=True, palette="Set2")
    plt.title("Pseudotime Violin Plot Across Conditions")
    plt.xlabel("Condition (Time Points)")
    plt.ylabel("Pseudotime")
    plt.legend(title="Response")
    plt.show()

plot_pseudotime_distribution(adata_valid)

# plot_pseudotime_umap
def plot_pseudotime_umap(adata):
    # scolorded as dpt_pseudotime
    sc.pl.umap(adata, color='dpt_pseudotime', title="Pseudotime Trajectory (All Conditions)")

    # group as Condition
    conditions = adata.obs['Condition'].unique()
    for condition in conditions:
        adata_condition = adata[adata.obs['time'] == condition].copy()
        sc.pl.umap(adata_condition, color='dpt_pseudotime', title=f"Pseudotime Trajectory: {condition}")

plot_pseudotime_umap(adata_valid)

# plot_condition_heatmap
def plot_condition_heatmap(adata):
    grouped_pseudotime = adata.obs.groupby(['time', 'Response'])['dpt_pseudotime'].mean().unstack()
    sns.heatmap(grouped_pseudotime, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Mean Pseudotime Across Conditions and Responses")
    plt.xlabel("Response")
    plt.ylabel("Condition")
    plt.show()

plot_condition_heatmap(adata_valid)
