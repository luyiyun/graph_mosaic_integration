import os.path as osp
from datetime import datetime
import logging

import numpy as np
import pandas as pd
import mudata as mu
import scanpy as sc
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import colorcet as cc
from scipy.optimize import linear_sum_assignment
from gmi import GraphMosaicIntegration
from gmi.metric import convert_mudata_to_anndata
from sklearn.preprocessing import normalize
from scib_metrics.benchmark import Benchmarker, BioConservation


def test_balance_weights():
    root = "./result/2024-12-20_20-33-"
    # root = "./result/2024-12-20_21-13"
    embed_with_umap_fn = osp.join(root, "embed_with_umap.h5mu")
    cutoff = 0.5
    power = 4

    if not osp.exists(embed_with_umap_fn):
        print(f"{embed_with_umap_fn} not found, generating...")
        embed_df = pd.read_csv(osp.join(root, "final_embeddings.csv"), index_col=0)
        embed = embed_df.values

        mdata = mu.read("./data/pbmc.h5mu")
        mdata.obs["batch"] = (
            mdata.mod["protein"]
            .obs["batch"]
            .loc[mdata.obs_names]
            .map(lambda x: f"batch{x}")
            .astype("category")
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

        mdata.obsm["gmi"] = embed[: mdata.n_obs]

        sc.pp.neighbors(mdata, use_rep="gmi")
        sc.tl.umap(mdata)
        mdata.write(embed_with_umap_fn)
    else:
        print(f"{embed_with_umap_fn} found, loading...")
        mdata = mu.read(embed_with_umap_fn)

    key = "cluster_per_batch"
    if key not in mdata.obs:
        print(f"{key} not found, calculating...")
        mdata.obs[key] = np.empty(mdata.n_obs, dtype=int)
        for batch in mdata.obs["batch"].unique():
            mask = mdata.obs["batch"] == batch
            embed_i = mdata.obsm["gmi"][mask]
            adata_i = sc.AnnData(X=embed_i, obs=mdata.obs.loc[mask])
            sc.pp.neighbors(
                adata_i, use_rep="X", n_pcs=embed_i.shape[1], metric="cosine"
            )
            sc.tl.leiden(adata_i, resolution=1.0)
            mdata.obs.loc[mask, key] = adata_i.obs["leiden"].cat.codes
        mdata.write(embed_with_umap_fn)

    key = "cluster_centers"
    if key not in mdata.uns:
        print(f"{key} not found, calculating...")
        centers, n_clusters = {}, {}
        for batch in mdata.obs["batch"].unique():
            mdata_i = mdata[mdata.obs["batch"] == batch]
            ui, ni = [], []
            for cluster_j in np.sort(mdata_i.obs["cluster_per_batch"].unique()):
                mask_j = mdata_i.obs["cluster_per_batch"] == cluster_j
                embed_mean_j = mdata_i.obsm["gmi"][mask_j].mean(axis=0)
                n_j = mask_j.sum()
                ui.append(embed_mean_j)
                ni.append(n_j)
            ui = np.stack(ui, axis=0)
            ui = normalize(ui, axis=1, norm="l2")
            ni = np.array(ni)
            centers[batch] = ui
            n_clusters[batch] = ni
        mdata.uns[key] = centers
        mdata.uns["n_clusters"] = n_clusters
        mdata.write(embed_with_umap_fn)

    fn = osp.join(root, "umap_cluster_per_batch.png")
    if not osp.exists(fn):
        colors = sns.color_palette()
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        axs = axs.flatten()
        for i, batch in enumerate(mdata.obs["batch"].unique()):
            ax = axs[i]
            mask = mdata.obs["batch"] == batch
            mdata_i = mdata[mask]
            for j, cluster_j in enumerate(mdata_i.obs["cluster_per_batch"].unique()):
                mask_j = mdata_i.obs["cluster_per_batch"] == cluster_j
                umap_xy = mdata_i.obsm["X_umap"][mask_j]
                umap_xy_mean = np.mean(umap_xy, axis=0)
                ax.plot(
                    mdata_i.obsm["X_umap"][mask_j, 0],
                    mdata_i.obsm["X_umap"][mask_j, 1],
                    ".",
                    label=f"cluster {cluster_j}",
                    color=colors[j],
                    markersize=3,
                    alpha=0.3,
                )
                ax.plot(
                    umap_xy_mean[0],
                    umap_xy_mean[1],
                    "o",
                    color=colors[j],
                    label=f"cluster {cluster_j}",
                )
        ax.legend()
        ax.set_title(f"batch {batch}")
        fig.tight_layout()
        fig.savefig(osp.join(root, "umap_cluster_per_batch.png"))

    key = "cluster_matching"
    if key not in mdata.uns:
        print(f"{key} not found, calculating...")
        matching = {}
        batches = mdata.obs["batch"].unique()
        for i, batch_i in enumerate(batches):
            ui = mdata.uns["cluster_centers"][batch_i]
            for j, batch_j in enumerate(batches[i + 1 :], start=i + 1):
                uj = mdata.uns["cluster_centers"][batch_j]
                cosine = ui @ uj.T
                cosine[cosine < cutoff] = 0
                cosine = np.power(cosine, power)
                matching[f"{batch_i}_{batch_j}"] = cosine
                print(f"matching {batch_i} and {batch_j}: ")
                print(cosine)
        mdata.uns[key] = matching
        mdata.write(embed_with_umap_fn)

    # 重新匹配聚类的标签
    for batch in mdata.obs["batch"].unique():
        key = f"cluster_per_batch_matched_by_{batch}"
        if key not in mdata.obs:
            mdata.obs[key] = mdata.obs["cluster_per_batch"].copy()
            for k, v in mdata.uns["cluster_matching"].items():
                k1, k2 = k.split("_")
                if k1 != batch and k2 != batch:
                    continue
                idx1, idx2 = linear_sum_assignment(v, maximize=True)
                if k1 == batch:
                    mask_i = mdata.obs["batch"] == k2
                    mdata.obs.loc[mask_i, key] = mdata.obs.loc[mask_i, key].map(
                        lambda x: {i2: i1 for i1, i2 in zip(idx1, idx2)}.get(x, np.nan)
                    )
                    print(f"convert {idx2} to {idx1} in {k2}")
                else:
                    mask_i = mdata.obs["batch"] == k1
                    mdata.obs.loc[mask_i, key] = mdata.obs.loc[mask_i, key].map(
                        lambda x: {i1: i2 for i1, i2 in zip(idx1, idx2)}.get(x, np.nan)
                    )
                    print(f"convert {idx1} to {idx2} in {k1}")

    fn = osp.join(root, "umap_cluster_per_batch_matched.png")
    if not osp.exists(fn):
        # colors = sns.color_palette()
        colors = cc.glasbey_light
        fig, axs = plt.subplots(4, 4, figsize=(12, 12))
        batches = mdata.obs["batch"].unique()
        for i, batch_i in enumerate(batches):
            mask = mdata.obs["batch"] == batch_i
            mdata_i = mdata[mask]
            for j, batch_j in enumerate(batches):
                cluster_ij = mdata_i.obs[f"cluster_per_batch_matched_by_{batch_j}"]
                ax = axs[j, i]
                cluster_ij_unique = cluster_ij.unique()
                cluster_ij_unique = np.sort(cluster_ij_unique)
                for k, cluster_k in enumerate(cluster_ij_unique):
                    mask_k = cluster_ij == cluster_k
                    umap_xy = mdata_i.obsm["X_umap"][mask_k]
                    ax.plot(
                        umap_xy[:, 0],
                        umap_xy[:, 1],
                        ".",
                        label=f"cluster {cluster_k}",
                        color="gray" if cluster_k == np.nan else colors[k],
                        markersize=3,
                    )
                # ax.legend()
                ax.set_title(f"b{batch_i[-1]} cells based on b{batch_j[-1]} clusters")
        fig.tight_layout()
        fig.savefig(osp.join(root, "umap_cluster_per_batch_matched.png"))

    # 按照glue的方式来计算样本得分
    key = "glue_balance_weights"
    if key not in mdata.obs:
        print(f"{key} not found, calculating...")
        batches = mdata.obs["batch"].unique()
        joint_cosine = 1.0
        for k, v in mdata.uns["cluster_matching"].items():
            k1, k2 = k.split("_")
            i1 = np.nonzero(batches == k1)[0]
            i2 = np.nonzero(batches == k2)[0]
            indice = tuple(
                slice(None) if k in (i1, i2) else np.newaxis
                for k in range(len(batches))
            )  # To align axes
            v = v[indice]
            joint_cosine = joint_cosine * v

        mdata.obs[key] = np.empty(mdata.n_obs, dtype=float)
        for i, b in enumerate(batches):
            mask = mdata.obs["batch"] == b
            mdata_i = mdata[mask]
            balancing = (
                joint_cosine.sum(
                    axis=tuple(k for k in range(joint_cosine.ndim) if k != i)
                )
                / mdata.uns["n_clusters"][b]
            )
            balancing = balancing[mdata_i.obs["cluster_per_batch"].values]
            balancing /= balancing.mean()
            mdata.obs.loc[mask, key] = balancing
        mdata.write(embed_with_umap_fn)

    fn = osp.join(root, "umap_glue_weights.png")
    if not osp.exists(fn):
        fig, axs = plt.subplots(ncols=2, figsize=(10, 6))
        batches = mdata.obs["batch"].unique()
        for i, batch in enumerate(batches):
            ax = axs[i]
            mask = mdata.obs["batch"] == batch
            ax.plot()
        fig, axs = plt.subplots(ncols=2, figsize=(10, 5))
        ax = axs[0]
        for batch in mdata.obs["batch"].unique():
            mdata_i = mdata[mdata.obs["batch"] == batch]
            ax.plot(
                mdata_i.obsm["X_umap"][:, 0],
                mdata_i.obsm["X_umap"][:, 1],
                ".",
                label=f"batch {batch}",
                markersize=3,
            )
        ax.legend()
        ax = axs[1]
        cb = ax.scatter(
            mdata.obsm["X_umap"][:, 0],
            mdata.obsm["X_umap"][:, 1],
            c=mdata.obs["glue_balance_weights"],
            cmap="coolwarm",
            s=3,
            alpha=1.0,
        )
        fig.colorbar(cb)
        fig.tight_layout()
        fig.savefig(osp.join(root, "umap_glue_weights.png"))

    # 计算新的balance weights
    key = "gmi_balance_weights"
    if key not in mdata.obsm:
        print(f"{key} not found, calculating...")
        batches = np.sort(mdata.obs["batch"].unique())
        print(batches)
        embed = mdata.obsm["gmi"]
        embed = normalize(embed, axis=1, norm="l2")
        balance_weights = np.zeros((mdata.n_obs, len(batches)))
        for i, bi in enumerate(batches):
            mask_i = mdata.obs["batch"].values == bi
            embed_i = embed[mask_i]
            embed_i_ = embed[~mask_i]
            cosine = embed_i @ embed_i_.T
            cosine[cosine < 0.1] = 0
            # cosine = np.power(cosine, power)
            scores_i = []
            for bj in batches:
                if bj == bi:
                    continue
                mask_j = mdata.obs["batch"].values[~mask_i] == bj
                scores_ij = cosine[:, mask_j].mean(axis=1)
                scores_i.append(scores_ij)
            scores_i = np.stack(scores_i, axis=1)
            scores_i = np.insert(scores_i, i, scores_i.max(axis=1), axis=1)
            balance_weights[mask_i] = scores_i
        balance_weights = balance_weights / balance_weights.mean()
        mdata.obsm[key] = balance_weights
        mdata.write(embed_with_umap_fn)

    # 绘制权重分布图
    fn = osp.join(root, "umap_gmi_weights.png")
    if not osp.exists(fn):
        key = "gmi_balance_weights"
        fig, axs = plt.subplots(
            ncols=4, nrows=4, figsize=(10, 10), layout="constrained"
        )
        batches = np.sort(mdata.obs["batch"].unique())
        vmin, vmax = mdata.obsm[key].min(), mdata.obsm[key].max()
        for i, batch_i in enumerate(batches):
            mask_i = mdata.obs["batch"] == batch_i
            mdata_i = mdata[mask_i]
            for j, batch_j in enumerate(batches):
                ax = axs[i, j]
                cm = ax.scatter(
                    mdata_i.obsm["X_umap"][:, 0],
                    mdata_i.obsm["X_umap"][:, 1],
                    c=mdata_i.obsm[key][:, j],
                    cmap="coolwarm",
                    s=3,
                    vmin=vmin,
                    vmax=vmax,
                )
                ax.set_title(f"b{batch_i[-1]} cells, b{batch_j[-1]} weights")
        fig.colorbar(cm, ax=axs.ravel().tolist())
        fig.savefig(fn)
        # NOTE: 不能单点计算，这样会导致在同一个簇中的点得到最高的权重，
        #   但是实际上这些点以及混合的非常好了，没有必要再进行提高权重

    # gmi_balance_weights_2
    # 结合glue和上面gmi_balance_weights的思路
    key = "gmi_balance_weights_2"
    # if key not in mdata.obsm:
    # print(f"{key} not found, calculating...")
    batches = np.sort(mdata.obs["batch"].unique())
    matches = mdata.uns["cluster_matching"]
    balance_weights = np.zeros((mdata.n_obs, len(batches)))
    for i, bi in enumerate(batches):
        scores_i = []
        for bj in batches:
            if bi == bj:
                continue
            if f"{bi}_{bj}" in matches:
                match_ij = matches[f"{bi}_{bj}"]
            elif f"{bj}_{bi}" in matches:
                match_ij = matches[f"{bj}_{bi}"].T
            else:
                raise ValueError(f"No matching found for {bi} and {bj}")
            match_ij = (
                match_ij
                / mdata.uns["n_clusters"][bj]
                / mdata.uns["n_clusters"][bi][:, None]
            )
            scores_ij = match_ij.sum(axis=1)
            scores_i.append(scores_ij)
        scores_i = np.stack(scores_i, axis=1)
        scores_i = np.insert(scores_i, i, scores_i.max(axis=1), axis=1)

        mask_i = mdata.obs["batch"].values == bi
        cluster_idx_i = mdata.obs.loc[mask_i, "cluster_per_batch"].values
        balance_weights[mask_i] = scores_i[cluster_idx_i]
    balance_weights = balance_weights / np.median(balance_weights)
    mdata.obsm[key] = balance_weights
    # print(balance_weights)
    mdata.write(embed_with_umap_fn)

    # 绘制权重分布图(2)
    fn = osp.join(root, "umap_gmi_weights_2.png")
    # if not osp.exists(fn):
    key = "gmi_balance_weights_2"
    fig, axs = plt.subplots(ncols=4, nrows=4, figsize=(10, 10), layout="constrained")
    batches = np.sort(mdata.obs["batch"].unique())
    vmin, vmax = mdata.obsm[key].min(), mdata.obsm[key].max()
    for i, batch_i in enumerate(batches):
        mask_i = mdata.obs["batch"] == batch_i
        mdata_i = mdata[mask_i]
        for j, batch_j in enumerate(batches):
            ax = axs[i, j]
            cm = ax.scatter(
                mdata_i.obsm["X_umap"][:, 0],
                mdata_i.obsm["X_umap"][:, 1],
                c=mdata_i.obsm[key][:, j],
                cmap="coolwarm",
                s=3,
                # vmin=vmin,
                # vmax=vmax,
                norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax),
            )
            ax.set_title(f"b{batch_i[-1]} cells, b{batch_j[-1]} weights")
    fig.colorbar(cm, ax=axs.ravel().tolist())
    fig.savefig(fn)


def main():
    logging.basicConfig(
        format="[%(name)s][%(asctime)s][%(levelname)s] %(message)s",
    )
    # logger = logging.getLogger("gmi.balance_weights")
    # logger.setLevel(logging.INFO)

    mdata = mu.read("./res/pbmc.h5mu")
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
    result_path = f"./res/{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
    # result_path = "./result/2024-12-21_15-59"

    if osp.exists(result_path):
        print(f"{result_path} already exists, read trained results...")
        mdata.obsm["gmi"] = (
            pd.read_csv(osp.join(result_path, "final_embeddings.csv"), index_col=0)
            .loc[mdata.obs_names, :]
            .values
        )
    else:
        # 设定参数
        gmi_model = GraphMosaicIntegration(
            label_smoothing=0.1,
            adversarial_training=True,
            adversarial_batching_method="divide",
            val_split=0.1,
            patience=10,
            num_epochs=200,
            device="cuda:0",
            adversartial_balance_weights=False,
            num_epochs_with_balanced_weights=20,
            learning_rate=0.005,
            add_batch_embedding=True,
            bilinear=True,
            w_grad_rev=0.2,
            w_loss_cls=0.2,
            w_cov=1e-3,
            # num_cluster=4,
            # w_loss_clu=0.0,
            # late_join_clu_weight=100,
        )
        gmi_model.fit(mdata, batch_key="batch", feature_interaction_key="net")
        gmi_model.save(result_path)
        gmi_model.plot_losses(osp.join(result_path, "losses.png"))

        # if gmi_model.adversartial_balance_weights:
        #     mdata.obs["weights"] = gmi_model.graph.nodes_adversarial_weights
        # else:
        #     mdata.obs["weights"] = gmi_model.trainer.estimate_balance_weights()
        #
        mdata.obsm["gmi"] = gmi_model.embeddings[: mdata.n_obs].detach().cpu().numpy()

        # fg = sns.displot(mdata.obs["weights"], kde=True, rug=True)
        # fg.savefig(osp.join(result_path, "weights_dist.png"))

        sc.pp.neighbors(mdata, use_rep="gmi")
        sc.tl.umap(mdata)
        fig = sc.pl.umap(
            mdata,
            color=["batch", "coarse_cluster", "cluster"],
            show=False,
            return_fig=True,
            ncols=3,
        )
        fig.savefig(osp.join(result_path, "umap.png"))

    print(mdata)
    adata = convert_mudata_to_anndata(
        mdata=mdata,
        sparse=True,
        fillna=0.0,
        obs=["coarse_cluster", "batch"],  # 指定保留的 obs 列
        obsm=["gmi"],  # 指定保留的 obsm 键
    )
    bm = Benchmarker(
        adata,
        batch_key="batch",
        label_key="coarse_cluster",
        embedding_obsm_keys=["gmi"],
        n_jobs=-1,
        bio_conservation_metrics=BioConservation(
            nmi_ari_cluster_labels_kmeans=False,
            nmi_ari_cluster_labels_leiden=True,
        ),
    )
    bm.benchmark()
    bm.plot_results_table(min_max_scale=False, save_dir=result_path)

    # 打印详细的结果数据框
    df = bm.get_results(min_max_scale=False)
    df_transposed = df.transpose()
    print(df_transposed)
    df_transposed.to_csv(osp.join(result_path, "benchmark_result.csv"))


if __name__ == "__main__":
    main()
    # a0 = 0.1
    # i = np.linspace(2, 40, num=100)
    # mul = 2 ** (1 / np.log(i)) ** 2
    # mul = np.r_[1.0, np.cumprod(mul)]
    # alpha = mul * a0
    #
    # plt.plot(np.arange(len(alpha)), alpha)
    # plt.xlabel("Number of Clusters")
    # plt.ylabel("Alpha")
    # plt.show()
