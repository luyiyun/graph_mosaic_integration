from datetime import datetime
import mudata as mu
import pandas as pd
from gmi import GraphMosaicIntegration,run_benchmark, data_infor_integrate,plot_umap


mdata_path = "/data/share_data/yuytest/gmi_data/pbmc.h5mu"
mdata = mu.read(mdata_path)
result_path = f"./result/{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
# 获取完整的batch和标签，并保存在batch和标签列
mdata = data_infor_integrate(mdata, feature_key="batch",saved_feature_name="batch", target_attr="obs")
mdata = data_infor_integrate(mdata, feature_key="coarse_cluster",saved_feature_name="label", target_attr="obs")
mdata = data_infor_integrate(mdata, feature_key="lsi_pca",saved_feature_name="Unintegrated", target_attr="obsm", dim_limit=100)

#设定参数
label_smoothing = 0.1
alpha = 0.1
loss_alpha = 0.2
neg_sampling_mode = "matched"



gmi_model = GraphMosaicIntegration(
    label_smoothing=label_smoothing,
    alpha=alpha,
    loss_alpha=loss_alpha,
    neg_sampling_mode=neg_sampling_mode,
    
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
    17055,
    result_path,
)
