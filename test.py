import os.path as osp
from datetime import datetime
import logging

import mudata as mu
import scanpy as sc
import numpy as np
import seaborn as sns
from gmi import GraphMosaicIntegration


logging.basicConfig(
    level=logging.WARNING,
    format="[%(name)s][%(asctime)s][%(levelname)s] %(message)s",
)


mdata = mu.read("./data/pbmc.h5mu")
mdata.obs["batch"] = (
    mdata.mod["protein"].obs["batch"].loc[mdata.obs_names].astype("category")
)
mdata.obs["coarse_cluster"] = (
    mdata.mod["protein"]
    .obs["coarse_cluster"]
    .loc[mdata.obs_names]
    .astype("category")
)
mdata.obs["cluster"] = (
    mdata.mod["protein"].obs["cluster"].loc[mdata.obs_names].astype("category")
)
result_path = f"./result/{datetime.now().strftime('%Y-%m-%d_%H-%M')}"

# 设定参数
label_smoothing = 0
alpha = 0.05
loss_alpha = 0.05
neg_sampling_mode = "matched"

gmi_model = GraphMosaicIntegration(
    label_smoothing=label_smoothing,
    alpha=alpha,
    loss_alpha=loss_alpha,
    neg_sampling_mode=neg_sampling_mode,
    neg_sample_in_batch=False,
    val_split=None,
    patience=np.inf,
    num_epochs=60,
)
gmi_model.fit(mdata, batch_key="batch", feature_interaction_key="net")
gmi_model.save(result_path)
weights = gmi_model.trainer.estimate_balance_weights()
mdata.obs["weights"] = weights
mdata.obsm["gmi"] = gmi_model.embeddings[: mdata.n_obs].detach().cpu().numpy()

fg = sns.displot(weights, kde=True, rug=True)
fg.savefig(osp.join(result_path, "weights_dist.png"))

sc.pp.neighbors(mdata, use_rep="gmi")
sc.tl.umap(mdata)
fig = sc.pl.umap(
    mdata,
    color=["batch", "coarse_cluster", "cluster", "weights"],
    show=False,
    return_fig=True,
    ncols=2,
)
fig.savefig(osp.join(result_path, "umap.png"))
