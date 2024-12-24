import os
import json
from datetime import datetime

import mudata as mu
from gmi import (
    GraphMosaicIntegration,
    run_benchmark,
    data_infor_integrate,
    plot_umap,
)

for i in [0]:
    mdata_path = "/data/share_data/yuytest/gmi_data/muto.h5mu"
    mdata = mu.read(mdata_path)
    result_path = f"./result/{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
    # 获取完整的batch和标签，并保存在batch和标签列
    mdata = data_infor_integrate(
        mdata,
        feature_key="batch",
        saved_feature_name="batch",
        target_attr="obs",
    )
    mdata = data_infor_integrate(
        mdata,
        feature_key="cell_type",
        saved_feature_name="label",
        target_attr="obs",
    )
    mdata = data_infor_integrate(
        mdata,
        feature_key="lsi_pca",
        saved_feature_name="Unintegrated",
        target_attr="obsm",
        dim_limit=100,
    )
    batch_mapping = {category: idx + 1 for idx, category in enumerate(mdata.obs['batch'].cat.categories)}
    mdata.obs['batch'] = mdata.obs['batch'].map(batch_mapping)
    # import ipdb; ipdb.set_trace()
    num_neg_per_pos=4
    label_smoothing =0.1
    alpha=0.05
    loss_alpha=i
    neg_sampling_mode = "matched"
    adversartial_balance_weights=False

    gmi_model = GraphMosaicIntegration(
        num_neg_per_pos=num_neg_per_pos,
        label_smoothing=0.1,
        alpha=0.05,
        loss_alpha=0.1,
        adversarial_training=True,
        adversarial_batching_method="divide",
        val_split=0.1,
        patience=5,
        num_epochs=100,
        late_join_alpha=0,
        late_join_loss_alpha=0,
        device="cuda:1",
        adversartial_balance_weights=False,
        num_epochs_with_balanced_weights=20,
        learning_rate=0.01,
        add_batch_embedding=True,
        bilinear=True,
    )
    gmi_model.fit(mdata, batch_key="batch", feature_interaction_key="net")
    gmi_model.save(result_path)

    plot_umap(
        mdata,
        result_path,
        neg_sampling_mode=neg_sampling_mode,
    )

    run_benchmark(
        mdata,
        mdata.obs.shape[0],
        result_path,
    )

params = {
    "label_smoothing": label_smoothing,
    "alpha": alpha,
    "loss_alpha": loss_alpha,
    "neg_sampling_mode": neg_sampling_mode,
    "num_neg_per_pos": num_neg_per_pos,
}

with open(os.path.join(result_path, "parameters.json"), "w") as f:
    json.dump(params, f, indent=4)
