import pandas as pd
import anndata as ad
import networkx as nx
import scanpy as sc
import mudata as mu
import scglue
from itertools import chain
from matplotlib import rcParams
from datetime import datetime
from gmi import (
    GraphMosaicIntegration,
    run_benchmark,
    data_infor_integrate,
    plot_umap,
)


mdata_path = "/data/share_data/yuytest/gmi_data/muto.h5mu"
mdata = mu.read(mdata_path)
result_path = f"./result/{datetime.now().strftime('%Y-%m-%d_%H-%M')}"


rna = mdata.mod['rna']
rna.layers["counts"] = rna.X.copy()
sc.pp.highly_variable_genes(rna, n_top_genes=2000, flavor="cell_ranger")
sc.pp.normalize_total(rna)
sc.pp.log1p(rna)
sc.pp.scale(rna)


atac = mdata.mod['atac']


guidance = scglue.genomics.rna_anchored_guidance_graph(rna, atac)

scglue.graph.check_graph(guidance, [rna, atac])

scglue.models.configure_dataset(
    rna, "NB", use_highly_variable=True,
    use_layer="counts", use_rep="lsi_pca"
)

scglue.models.configure_dataset(
    atac, "NB", use_highly_variable=True,
    use_rep="lsi_pca"
)



guidance_hvf = guidance.subgraph(chain(
    rna.var.query("highly_variable").index,
    atac.var.query("highly_variable").index
)).copy()
#igraph

glue = scglue.models.fit_SCGLUE(
    {"rna": rna, "atac": atac}, guidance,
    fit_kws={"directory": "glue_full"}
)

glue.save("glue.dill")

rna.obsm["X_glue"] = glue.encode_data("rna", rna)
atac.obsm["X_glue"] = glue.encode_data("atac", atac)
combined = ad.concat([rna, atac])

sc.pp.neighbors(combined, use_rep="X_glue", metric="cosine")

feature_embeddings = glue.encode_graph(guidance_hvf)
feature_embeddings = pd.DataFrame(feature_embeddings, index=glue.vertices)
feature_embeddings.iloc[:5, :5]
rna.varm["X_glue"] = feature_embeddings.reindex(rna.var_names).to_numpy()
atac.varm["X_glue"] = feature_embeddings.reindex(atac.var_names).to_numpy()


combined.write(os.path.join(result_path, "glue_muto_embedding.h5ad"))



