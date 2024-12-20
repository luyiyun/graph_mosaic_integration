import os.path as osp
from datetime import datetime
import logging

import mudata as mu
import scanpy as sc
import seaborn as sns
from gmi import GraphMosaicIntegration


logging.basicConfig(
    format="[%(name)s][%(asctime)s][%(levelname)s] %(message)s",
)
logger = logging.getLogger("gmi.balance_weights")
logger.setLevel(logging.INFO)


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
gmi_model = GraphMosaicIntegration(
    label_smoothing=0.1,
    alpha=0.1,
    loss_alpha=0.1,
    adversarial_training=True,
    adversarial_batching_method="divide",
    val_split=0.1,
    patience=5,
    num_epochs=100,
    late_join_alpha=0,
    late_join_loss_alpha=0,
    device="cuda:0",
    adversartial_balance_weights=False,
    num_epochs_with_balanced_weights=20,
    learning_rate=0.01,
)
gmi_model.fit(mdata, batch_key="batch", feature_interaction_key="net")
gmi_model.save(result_path)
gmi_model.plot_losses(osp.join(result_path, "losses.png"))

if gmi_model.adversartial_balance_weights:
    mdata.obs["weights"] = gmi_model.graph.nodes_adversarial_weights
else:
    mdata.obs["weights"] = gmi_model.trainer.estimate_balance_weights()

mdata.obsm["gmi"] = (
    gmi_model.embeddings[: mdata.n_obs].detach().cpu().numpy()
)

fg = sns.displot(mdata.obs["weights"], kde=True, rug=True)
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
