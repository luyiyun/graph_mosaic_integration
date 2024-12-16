from datetime import datetime
from dataclasses import asdict

# import os
# import torch
# import seaborn as sns
# import numpy as np
# import pandas as pd
# import scanpy as sc
import mudata as mu

# import matplotlib.pyplot as plt
# from gmi import Trainer, run_benchmark
from gmi import GraphMosaicIntegration

# File paths
# NODES_PATH = "/home/yuyipei/graph_mosaic_integration/data/All_nodes.csv"
# WEIGHTS_PATH = "/home/yuyipei/graph_mosaic_integration/data/Weights.csv"
# mdata_path = "/data/share_data/yuytest/gmi_data/MOP.h5mu"
# Save file paths

# edge_path = "./data/edges.pt"
# edge_weights_path = "./data/edge_weights.pt"
# device = "cuda:1" if torch.cuda.is_available() else "cpu"


# if __name__ == "__main__":
#     mdata_path = "/data/share_data/yuytest/gmi_data/MOP.h5mu"
#     for i in ['non_domain']:
#         result = f"/home/yuyipei/graph_mosaic_integration/result/{i}"
#         train_model(i,result)
#         evaluate_trained_embeddings(i,result,mdata_path)
#         plot_ablation()

mdata = mu.read("./data/pbmc.h5mu")
mdata.obs["batch"] = mdata.mod["protein"].obs["batch"].loc[mdata.obs_names]
gmi_model = GraphMosaicIntegration(num_epochs=2)
gmi_model.fit(mdata, batch_key="batch", feature_interaction_key="net")
print(gmi_model.embeddings)
save_path = f"./result/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
gmi_model.save(save_path)

new_model = GraphMosaicIntegration.load(save_path)
print(new_model.embeddings)

# graph = GraphMosaicIntegration.mdata2graph(
#     mdata, batch_key="batch", feature_interaction_key="net"
# )
# graph.save("./data/graph.h5")
# graph = MosaicDataGraph.load("./data/graph.h5")
# print(graph)
