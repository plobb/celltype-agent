"""Demo: automated cell type annotation with celltype-agent.

Run this as a plain Python script, or convert to a Jupyter notebook with:

    pip install jupytext
    jupytext --to notebook notebooks/demo.py

Each "# %%" marker is a notebook cell boundary when opened with Jupytext or
the VS Code Jupyter extension.
"""

# %% [markdown]
# # celltype-agent demo
#
# Automated cell type annotation for single-cell RNA-seq data using Claude.
#
# **Prerequisites**
# ```bash
# pip install celltype-agent[scanpy]
# export ANTHROPIC_API_KEY=sk-ant-...
# ```

# %% [markdown]
# ## 1. Load PBMC3K

# %%
import warnings

warnings.filterwarnings("ignore")

import scanpy as sc

sc.settings.verbosity = 1

# Download and preprocess the classic 3k PBMC dataset.
# scanpy caches it locally so subsequent runs are instant.
adata = sc.datasets.pbmc3k_processed()
print(adata)

# %% [markdown]
# ## 2. Annotate clusters

# %%
import logging

# Basic logging so progress messages appear in the notebook output
logging.basicConfig(level=logging.INFO, format="%(message)s")

from celltype_agent import annotate

messages = []

result = annotate(
    adata,
    species="human",
    tissue="PBMC",
    cluster_key="louvain",     # PBMC3K uses louvain clusters
    n_markers=10,
    progress_callback=messages.append,
)

# %% [markdown]
# ## 3. Rich HTML summary (auto-rendered in Jupyter)

# %%
result   # triggers _repr_html_ in Jupyter; prints repr in script mode

# %% [markdown]
# ## 4. Annotations as a DataFrame

# %%
df = result.to_dataframe()
df

# %% [markdown]
# ## 5. Save to CSV

# %%
result.to_csv("pbmc3k_annotations.csv")
print("Saved to pbmc3k_annotations.csv")

# %% [markdown]
# ## 6. Inspect the annotated AnnData

# %%
# annotate() wrote labels back to adata.obs["cell_type"] by default
print(adata.obs[["louvain", "cell_type"]].drop_duplicates().sort_values("louvain"))

# %% [markdown]
# ## 7. UMAP coloured by predicted cell type

# %%
sc.pl.umap(
    adata,
    color=["louvain", "cell_type"],
    legend_loc="on data",
    frameon=False,
    title=["Louvain cluster", "Predicted cell type"],
)
