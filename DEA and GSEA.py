#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install --upgrade pip')
get_ipython().system('pip install --upgrade setuptools wheel')
get_ipython().system('pip install --upgrade "mxnet<2.0.0"')
get_ipython().system('pip install autogluon')


# In[3]:


get_ipython().system('pip install pydeseq2')
get_ipython().system('pip install adjustText')
get_ipython().system('pip install scanpy')


# In[4]:


pip install igraph


# In[5]:


pip install leidenalg


# In[6]:


import sys
print(sys.executable)


# In[7]:


import sys
get_ipython().system('{sys.executable} -m pip install scanpy')


# In[8]:


import scanpy as sc
import pandas as pd

expression_matrix = pd.read_csv("GSE166181_Normalized_UMI_CountMatrix.tsv", sep="\t", index_col=0)


# In[9]:

meta_data = pd.read_csv("GSE166181_Metadata.tsv", sep="\t", index_col=0)


# In[10]:


adata = sc.AnnData(X=expression_matrix.T)  
adata.obs = meta_data


# In[11]:


# # QC
# sc.pp.filter_cells(adata, min_genes=200)
# sc.pp.filter_genes(adata, min_cells=3)


# In[12]:


# # 
# adata.var["mt"] = adata.var_names.str.startswith("MT-")
# sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, inplace=True)


# In[13]:

# adata = adata[adata.obs["pct_counts_mt"] < 5, :].copy()


# In[14]:


# norm
# sc.pp.normalize_total(adata, target_sum=1e4)
# sc.pp.log1p(adata)


# In[15]:


sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
adata = adata[:, adata.var.highly_variable]


# In[16]:


highly_variable_genes = adata.var[adata.var['highly_variable']].index.tolist()

print(f"Number of highly variable genes: {len(highly_variable_genes)}")
print("Highly variable genes:", highly_variable_genes[:10])  


# In[17]:


adata = adata[:, adata.var.highly_variable].copy()

sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata, svd_solver="arpack")


# In[18]:


sc.pl.pca_variance_ratio(adata, log=True)

# In[19]:


# UMAP
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)  
sc.tl.umap(adata)


# In[20]:


#adata.obs['Condition'] = adata.obs['time']


sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)

sc.tl.leiden(adata, resolution=0.8, flavor="igraph", n_iterations=2, directed=False)

# UMAP 
sc.tl.umap(adata)
sc.pl.umap(adata, color=['leiden', 'time'], s=50)


# In[21]:


# 按聚类可视化
sc.pl.umap(adata, color='leiden', title='UMAP colored by Leiden clusters')

# 按条件可视化
sc.pl.umap(adata, color='time', title='UMAP colored by Condition')


# In[22]:


print(adata.obs['time'].unique())


# In[23]:

filtered_conditions = ['R T0',  'R T1', 'R T2']
adata_filtered = adata[adata.obs['time'].isin(filtered_conditions)].copy()
print(adata_filtered.obs['time'].value_counts())
sc.pl.umap(adata_filtered, color='time', title='UMAP: T0, T1, T2', s=50)


# In[24]:

filtered_conditions = ['NR T0',  'NR T1', 'NR T2']
adata_filtered = adata[adata.obs['time'].isin(filtered_conditions)].copy()
print(adata_filtered.obs['time'].value_counts())

sc.pl.umap(adata_filtered, color='time', title='UMAP: T0, T1, T2', s=50)


# In[25]:

sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon')

clusters = adata.obs['leiden'].unique().tolist()
print("All clusters:", clusters)

for cluster in clusters:
    print(f"Top genes for cluster {cluster}:")
    print(adata.uns['rank_genes_groups']['names'][cluster][:10])  # 输出前 10 个特异性基因
    print()


# In[26]:

mait_clusters = ['6']  # cluster 6 MAIT cells
mait_adata = adata[adata.obs['leiden'].isin(mait_clusters)].copy()

print(mait_adata.obs['time'].value_counts())

sc.pl.umap(mait_adata, color='time', title="MAIT Cells in Different Conditions")


# In[27]:

filtered_conditions = ['R T0', 'R T1', 'R T2']
adata_mait_R = mait_adata[mait_adata.obs['time'].isin(filtered_conditions)].copy()

print(adata_mait_R.obs['time'].value_counts())

sc.pl.umap(adata_mait_R, color='time', title="MAIT Cells in R Conditions")


# In[28]:

sc.tl.leiden(adata_mait_R, resolution=0.3)  
sc.pl.umap(adata_mait_R, color='leiden', title="MAIT Cell Subclusters")



# In[29]:

subcluster_counts = adata_mait_R.obs['leiden'].value_counts()
print(subcluster_counts)

valid_subclusters = subcluster_counts[subcluster_counts > 10].index
print(f"Valid subclusters: {valid_subclusters}")


# In[30]:


#Valid subclusters: ['0', '1', '2']

adata_valid = adata_mait_R[adata_mait_R.obs['leiden'].isin(valid_subclusters)].copy()
sc.tl.rank_genes_groups(adata_valid, 'leiden', method='wilcoxon')
sc.pl.rank_genes_groups(adata_valid, n_genes=10, sharey=False)


# In[31]:

print(adata_valid.obs.columns)


# In[32]:



adata_valid.obs['subcluster_type'] = adata_valid.obs['leiden'].map({
    '0': ' MAIT 0',  
    '1': ' MAIT 1',
    '2': ' MAIT 2',
    
})


print(adata_valid.obs[['leiden', 'subcluster_type']].head())


# In[33]:


time_distribution = adata_valid.obs.groupby(['subcluster_type', 'time']).size().unstack(fill_value=0)

print(time_distribution)

time_distribution.plot(kind='bar', stacked=True, title="MAIT Subclusters Distribution Across T0, T1, T2", ylabel="Cell Count", figsize=(10, 6))


# In[34]:

sc.pl.umap(adata_valid, color=['time', 'subcluster_type'],
           title="MAIT Subclusters Distribution Across T0, T1, T2")


# In[89]:


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

result_NR_T0 = sc.get.rank_genes_groups_df(adata_T0, group='NR')

result_NR_T0['log10_pval'] = -np.log10(result_NR_T0['pvals'])


result_NR_T0 = result_NR_T0.dropna(subset=['logfoldchanges', 'log10_pval'])
result_NR_T0 = result_NR_T0[result_NR_T0['log10_pval'] > 0]  

# volcano
plt.figure(figsize=(10, 6))
sns.scatterplot(data=result_NR_T0, x='logfoldchanges', y='log10_pval', hue='names', palette='coolwarm', legend=False)
top_genes = result_NR_T0.nlargest(10, 'log10_pval')


for i in range(len(top_genes)):
    plt.text(
        top_genes.iloc[i]['logfoldchanges'], 
        top_genes.iloc[i]['log10_pval'],
        top_genes.iloc[i]['names'], 
        fontsize=9, 
        ha='right', 
        color='black',
        va='bottom' 
    )


plt.axvline(x=0, linestyle='--', color='gray', linewidth=1)


plt.title("Volcano Plot for NR at T0", fontsize=16)
plt.xlabel('Log Fold Change', fontsize=14)
plt.ylabel('-log10(p-value)', fontsize=14)


plt.show()


print("Top 10 Significant Genes for NR at T0:")
print(top_genes[['names', 'logfoldchanges', 'log10_pval']])


# In[90]:


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

#  NR T1 
result_NR_T1 = sc.get.rank_genes_groups_df(adata_T1, group='NR')

result_NR_T1['log10_pval'] = -np.log10(result_NR_T1['pvals'])


result_NR_T1 = result_NR_T1.dropna(subset=['logfoldchanges', 'log10_pval'])
result_NR_T1 = result_NR_T1[result_NR_T1['log10_pval'] > 0]  # 移除 p-value 为零或负值的行

plt.figure(figsize=(10, 6))
sns.scatterplot(data=result_NR_T1, x='logfoldchanges', y='log10_pval', hue='names', palette='coolwarm', legend=False)

top_genes = result_NR_T1.nlargest(10, 'log10_pval')


for i in range(len(top_genes)):
    plt.text(
        top_genes.iloc[i]['logfoldchanges'], 
        top_genes.iloc[i]['log10_pval'],
        top_genes.iloc[i]['names'], 
        fontsize=9, 
        ha='right', 
        color='black',
        va='bottom'  # 让文字在点的下方显示，避免与点重叠
    )


plt.axvline(x=0, linestyle='--', color='gray', linewidth=1)


plt.title("Volcano Plot for NR at T1", fontsize=16)
plt.xlabel('Log Fold Change', fontsize=14)
plt.ylabel('-log10(p-value)', fontsize=14)


plt.show()


print("Top 10 Significant Genes for NR at T1:")
print(top_genes[['names', 'logfoldchanges', 'log10_pval']])


# In[91]:


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# NR T2 
result_NR_T2 = sc.get.rank_genes_groups_df(adata_T2, group='NR')

result_NR_T2['log10_pval'] = -np.log10(result_NR_T2['pvals'])


result_NR_T2 = result_NR_T2.dropna(subset=['logfoldchanges', 'log10_pval'])
result_NR_T2 = result_NR_T2[result_NR_T2['log10_pval'] > 0]  # 移除 p-value 为零或负值的行


plt.figure(figsize=(10, 6))


sns.scatterplot(data=result_NR_T2, x='logfoldchanges', y='log10_pval', hue='names', palette='coolwarm', legend=False)


top_genes = result_NR_T2.nlargest(10, 'log10_pval')


for i in range(len(top_genes)):
    plt.text(
        top_genes.iloc[i]['logfoldchanges'], 
        top_genes.iloc[i]['log10_pval'],
        top_genes.iloc[i]['names'], 
        fontsize=9, 
        ha='right', 
        color='black',
        va='bottom'  
    )

plt.axvline(x=0, linestyle='--', color='gray', linewidth=1)


plt.title("Volcano Plot for NR at T2", fontsize=16)
plt.xlabel('Log Fold Change', fontsize=14)
plt.ylabel('-log10(p-value)', fontsize=14)


plt.show()


print("Top 10 Significant Genes for NR at T2:")
print(top_genes[['names', 'logfoldchanges', 'log10_pval']])


# In[102]:


import scanpy as sc
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


adata_T0 = adata[adata.obs['TimePoint'] == 'T0'].copy()


if 'R' not in adata_T0.obs['MainCondition'].unique():
    print("Warning: No 'R' condition data for Time Point T0")
if 'NR' not in adata_T0.obs['MainCondition'].unique():
    print("Warning: No 'NR' condition data for Time Point T0")


adata_R_T0 = adata_T0[adata_T0.obs['MainCondition'] == 'R'].copy()


adata_NR_T0 = adata_T0[adata_T0.obs['MainCondition'] == 'NR'].copy()


if adata_R_T0.shape[0] > 0 and adata_NR_T0.shape[0] > 0:
    
    sc.tl.rank_genes_groups(adata_T0, groupby='MainCondition', reference='NR', method='wilcoxon')

   
    result_R_T0 = sc.get.rank_genes_groups_df(adata_T0, group='R')
    
    
    result_R_T0['log10_pval'] = -np.log10(result_R_T0['pvals_adj'])  

    
    result_R_T0 = result_R_T0.dropna(subset=['logfoldchanges', 'log10_pval'])
    result_R_T0 = result_R_T0[result_R_T0['log10_pval'] > 0] 

    # volcano
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=result_R_T0, x='logfoldchanges', y='log10_pval', hue='names', palette='coolwarm', legend=False)
    top_genes = result_R_T0.nlargest(10, 'log10_pval')


    for i in range(len(top_genes)):
        plt.text(
            top_genes.iloc[i]['logfoldchanges'], 
            top_genes.iloc[i]['log10_pval'],
            top_genes.iloc[i]['names'], 
            fontsize=9, 
            ha='right', 
            color='black',
            va='bottom' 
        )

    plt.axvline(x=0, linestyle='--', color='gray', linewidth=1)


    plt.title("Volcano Plot for R at T0", fontsize=16)
    plt.xlabel('Log Fold Change', fontsize=14)
    plt.ylabel('-log10(p-value)', fontsize=14)

 
    plt.show()


    print("Top 10 Significant Genes for R at T0:")
    print(top_genes[['names', 'logfoldchanges', 'log10_pval']])
else:
    print("Not enough data for R and NR comparison at T0.")


# In[103]:


import scanpy as sc
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


adata_T1 = adata[adata.obs['TimePoint'] == 'T1'].copy()

if 'R' not in adata_T1.obs['MainCondition'].unique():
    print("Warning: No 'R' condition data for Time Point T1")
if 'NR' not in adata_T1.obs['MainCondition'].unique():
    print("Warning: No 'NR' condition data for Time Point T1")

adata_R_T1 = adata_T1[adata_T1.obs['MainCondition'] == 'R'].copy()

adata_NR_T1 = adata_T1[adata_T1.obs['MainCondition'] == 'NR'].copy()

if adata_R_T1.shape[0] > 0 and adata_NR_T1.shape[0] > 0:
    
    sc.tl.rank_genes_groups(adata_T1, groupby='MainCondition', reference='NR', method='wilcoxon')

    result_R_T1 = sc.get.rank_genes_groups_df(adata_T1, group='R')

    result_R_T1['log10_pval'] = -np.log10(result_R_T1['pvals_adj'])  

    result_R_T1 = result_R_T1.dropna(subset=['logfoldchanges', 'log10_pval'])
    result_R_T1 = result_R_T1[result_R_T1['log10_pval'] > 0]  

    # volcano
    plt.figure(figsize=(10, 6))

    sns.scatterplot(data=result_R_T1, x='logfoldchanges', y='log10_pval', hue='names', palette='coolwarm', legend=False)

    top_genes = result_R_T1.nlargest(10, 'log10_pval')


    for i in range(len(top_genes)):
        plt.text(
            top_genes.iloc[i]['logfoldchanges'], 
            top_genes.iloc[i]['log10_pval'],
            top_genes.iloc[i]['names'], 
            fontsize=9, 
            ha='right', 
            color='black',
            va='bottom'  
        )


    plt.axvline(x=0, linestyle='--', color='gray', linewidth=1)


    plt.title("Volcano Plot for R at T1", fontsize=16)
    plt.xlabel('Log Fold Change', fontsize=14)
    plt.ylabel('-log10(p-value)', fontsize=14)

  
    plt.show()

  
    print("Top 10 Significant Genes for R at T1:")
    print(top_genes[['names', 'logfoldchanges', 'log10_pval']])
else:
    print("Not enough data for R and NR comparison at T1.")


# In[104]:


import scanpy as sc
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


adata_T2 = adata[adata.obs['TimePoint'] == 'T2'].copy()

if 'R' not in adata_T2.obs['MainCondition'].unique():
    print("Warning: No 'R' condition data for Time Point T2")
if 'NR' not in adata_T2.obs['MainCondition'].unique():
    print("Warning: No 'NR' condition data for Time Point T2")


adata_R_T2 = adata_T2[adata_T2.obs['MainCondition'] == 'R'].copy()


adata_NR_T2 = adata_T2[adata_T2.obs['MainCondition'] == 'NR'].copy()


if adata_R_T2.shape[0] > 0 and adata_NR_T2.shape[0] > 0:
 
    sc.tl.rank_genes_groups(adata_T2, groupby='MainCondition', reference='NR', method='wilcoxon')


    result_R_T2 = sc.get.rank_genes_groups_df(adata_T2, group='R')

    result_R_T2['log10_pval'] = -np.log10(result_R_T2['pvals_adj'])  

    result_R_T2 = result_R_T2.dropna(subset=['logfoldchanges', 'log10_pval'])
    result_R_T2 = result_R_T2[result_R_T2['log10_pval'] > 0]  


    plt.figure(figsize=(10, 6))

    sns.scatterplot(data=result_R_T2, x='logfoldchanges', y='log10_pval', hue='names', palette='coolwarm', legend=False)

  
    top_genes = result_R_T2.nlargest(10, 'log10_pval')


    for i in range(len(top_genes)):
        plt.text(
            top_genes.iloc[i]['logfoldchanges'], 
            top_genes.iloc[i]['log10_pval'],
            top_genes.iloc[i]['names'], 
            fontsize=9, 
            ha='right', 
            color='black',
            va='bottom'  
        )


    plt.axvline(x=0, linestyle='--', color='gray', linewidth=1)

    plt.title("Volcano Plot for R at T2", fontsize=16)
    plt.xlabel('Log Fold Change', fontsize=14)
    plt.ylabel('-log10(p-value)', fontsize=14)

    plt.show()

    print("Top 10 Significant Genes for R at T2:")
    print(top_genes[['names', 'logfoldchanges', 'log10_pval']])
else:
    print("Not enough data for R and NR comparison at T2.")


# In[ ]:





# In[105]:


pip install gseapy


# In[134]:


import gseapy as gp
import pandas as pd
import numpy as np


result_R_T0['log10_pval'] = -np.log10(result_R_T0['pvals_adj'])  
result_R_T0 = result_R_T0.dropna(subset=['logfoldchanges', 'log10_pval'])


ranked_genes_R_T0 = result_R_T0[['names', 'logfoldchanges']].set_index('names')
ranked_genes_R_T0 = ranked_genes_R_T0.sort_values(by='logfoldchanges', ascending=False)


gsea_results_R_T0 = gp.prerank(
    rnk=ranked_genes_R_T0,
    gene_sets='KEGG_2021_Human',  
    permutation_num=1000,  
    min_size=15, 
    max_size=500,  
    outdir='./gsea_R_T0'  
)


print(gsea_results_R_T0.res2d.head(10))  


# In[145]:

top_terms_R_T0 = gsea_results_R_T0.res2d['Term'].head(10).tolist()
print("Top 10 terms:", top_terms_R_T0)

gsea_results_R_T0.plot(
    terms=top_terms, 
    figsize=(10, 6)
)

plt.show()


# In[136]:


import gseapy as gp
import pandas as pd
import numpy as np



result_R_T1['log10_pval'] = -np.log10(result_R_T1['pvals_adj'])  
result_R_T1 = result_R_T1.dropna(subset=['logfoldchanges', 'log10_pval'])


ranked_genes_R_T1 = result_R_T1[['names', 'logfoldchanges']].set_index('names')
ranked_genes_R_T1 = ranked_genes_R_T1.sort_values(by='logfoldchanges', ascending=False)


gsea_results_R_T1 = gp.prerank(
    rnk=ranked_genes_R_T1,
    gene_sets='KEGG_2021_Human',  
    permutation_num=1000,  
    min_size=15,  
    max_size=500,  
    outdir='./gsea_R_T1'  
)


print(gsea_results_R_T1.res2d.head(10)) 


# In[144]:



top_terms_R_T1 = gsea_results_R_T1.res2d['Term'].head(10).tolist()
print("Top 10 terms:", top_terms_R_T1)


gsea_results_R_T1.plot(
    terms=top_terms_R_T1,  
    figsize=(10, 6)
)


plt.show()


# In[146]:


import gseapy as gp
import pandas as pd
import numpy as np


result_R_T2['log10_pval'] = -np.log10(result_R_T2['pvals_adj'])  
result_R_T2 = result_R_T2.dropna(subset=['logfoldchanges', 'log10_pval'])


ranked_genes_R_T2 = result_R_T2[['names', 'logfoldchanges']].set_index('names')
ranked_genes_R_T2 = ranked_genes_R_T2.sort_values(by='logfoldchanges', ascending=False)


gsea_results_R_T2 = gp.prerank(
    rnk=ranked_genes_R_T2,
    gene_sets='KEGG_2021_Human',  
    permutation_num=1000,  
    min_size=15,  
    max_size=500,  
    outdir='./gsea_R_T2'  
)


print(gsea_results_R_T2.res2d.head(10))  


# In[147]:



top_terms_R_T2 = gsea_results_R_T2.res2d['Term'].head(10).tolist()
print("Top 10 terms:", top_terms_R_T2)


gsea_results_R_T2.plot(
    terms=top_terms_R_T2,  
    figsize=(10, 6)
)


plt.show()


# In[148]:


import gseapy as gp
import pandas as pd
import numpy as np




result_NR_T0['log10_pval'] = -np.log10(result_NR_T0['pvals_adj'])  
result_NR_T0 = result_NR_T0.dropna(subset=['logfoldchanges', 'log10_pval'])


ranked_genes_NR_T0 = result_NR_T0[['names', 'logfoldchanges']].set_index('names')
ranked_genes_NR_T0 = ranked_genes_NR_T0.sort_values(by='logfoldchanges', ascending=False)

# GSEA
gsea_results_NR_T0 = gp.prerank(
    rnk=ranked_genes_NR_T0,
    gene_sets='KEGG_2021_Human',  
    permutation_num=1000,  
    min_size=15,  
    max_size=500,  
    outdir='./gsea_NR_T0'  
)


print(gsea_results_NR_T0.res2d.head(10))  


# In[149]:

top_terms_NR_T0 = gsea_results_NR_T0.res2d['Term'].head(10).tolist()
print("Top 10 terms:", top_terms_NR_T0)


gsea_results_NR_T0.plot(
    terms=top_terms_NR_T0,  
    figsize=(10, 6)
)


plt.show()


# In[150]:


import gseapy as gp
import pandas as pd
import numpy as np




result_NR_T1['log10_pval'] = -np.log10(result_NR_T1['pvals_adj']) 
result_NR_T1 = result_NR_T1.dropna(subset=['logfoldchanges', 'log10_pval'])


ranked_genes_NR_T1 = result_NR_T1[['names', 'logfoldchanges']].set_index('names')
ranked_genes_NR_T1 = ranked_genes_NR_T1.sort_values(by='logfoldchanges', ascending=False)


gsea_results_NR_T1 = gp.prerank(
    rnk=ranked_genes_NR_T1,
    gene_sets='KEGG_2021_Human',  
    permutation_num=1000,  
    min_size=15,  
    max_size=500,  
    outdir='./gsea_NR_T1'  
)


print(gsea_results_NR_T1.res2d.head(10))  


# In[151]:


top_terms_NR_T1 = gsea_results_NR_T1.res2d['Term'].head(10).tolist()
print("Top 10 terms:", top_terms_NR_T1)


gsea_results_NR_T1.plot(
    terms=top_terms_NR_T1,  
    figsize=(10, 6)
)


plt.show()


# In[152]:


import gseapy as gp
import pandas as pd
import numpy as np




result_NR_T2['log10_pval'] = -np.log10(result_NR_T2['pvals_adj'])  
result_NR_T2 = result_NR_T2.dropna(subset=['logfoldchanges', 'log10_pval'])


ranked_genes_NR_T2 = result_NR_T2[['names', 'logfoldchanges']].set_index('names')
ranked_genes_NR_T2 = ranked_genes_NR_T2.sort_values(by='logfoldchanges', ascending=False)


gsea_results_NR_T2 = gp.prerank(
    rnk=ranked_genes_NR_T2,
    gene_sets='KEGG_2021_Human',  
    permutation_num=1000, 
    min_size=15,  
    max_size=500,  
    outdir='./gsea_NR_T2'  
)


print(gsea_results_NR_T2.res2d.head(10))  


# In[153]:



top_terms_NR_T1 = gsea_results_NR_T1.res2d['Term'].head(10).tolist()
print("Top 10 terms:", top_terms_NR_T1)


gsea_results_NR_T1.plot(
    terms=top_terms_NR_T1,  
    figsize=(10, 6)
)

plt.show()







