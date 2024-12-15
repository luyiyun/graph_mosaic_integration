
def train_model(i,result_dir):

    mdata = mu.read(mdata_path)

    print(mdata)
    weights = pd.read_csv(WEIGHTS_PATH, index_col=0)
    src_nodes = torch.tensor(weights["batch"].values, dtype=torch.long)
    dst_nodes = torch.tensor(weights["feature"].values, dtype=torch.long)
    edges = torch.stack([src_nodes, dst_nodes], dim=1)
    edge_weights = torch.tensor(
        weights["expression"].astype(float).values, dtype=torch.float
    )

    # Save tensors to disk
    torch.save(edges, edge_path)
    torch.save(edge_weights, edge_weights_path)

    net = mdata.varp["net"]
    coords = net.nonzero()
    values = net.data

    nodes = pd.read_csv(NODES_PATH, index_col=0)
    sample_edges = edges
    sample_edge_weights = edge_weights
    num_nodes = (
        max(src_nodes.max().item(), dst_nodes.max().item()) + 1
    )  # 更新 num_nodes（确保涵盖样本中所有节点的索引范围）
    num_cell = src_nodes.max().item() + 1
    num_batch = 5
    num_edges = sample_edges.size(0)

    # 提取特征间网络

    feat_net = torch.stack(
        [
            torch.tensor(coords[0] + num_cell, dtype=torch.long),  # 行索引
            torch.tensor(coords[1] + num_cell, dtype=torch.long),  # 列索引
        ],
        dim=1,
    )
    feat_weight = torch.tensor(values, dtype=torch.float64)  # 值

    # rms prop
    # 参数设置
    embedding_dim = 50
    num_neg_per_pos = 5
    learning_rate = 0.01
    num_epochs = 100
    batch_size = 131072
    val_split = 0.2  # 验证集比例
    neg_sampling_mode = "matched"
    add_feature_net = True
    neg_sample_in_batch = False
    num_domains = weights["batch"].nunique()
    # 初始化训练器
    trainer = GMI(
        optimizer="adam",
        num_nodes=num_nodes,
        num_cell=num_cell,
        embedding_dim=embedding_dim,
        label_smoothing=0,
        bn=True,
        num_neg_samples=num_neg_per_pos,
        neg_sampling_mode=neg_sampling_mode,
        neg_sample_in_batch=neg_sample_in_batch,
        batch_size=batch_size,
        lr=learning_rate,  # LR scater
        loss_type="weighted_softmax",
        device=device,
        add_batch_embedding=False,
        num_batch=num_batch,
        add_feature_net=add_feature_net,
        alpha=0.3,
        loss_alpha=0.2,
        batch_removal_method="divide",
        late_join_alpha=5,
        late_join_loss_alpha=5,
        patience=10,
        neg_sampling_restrict=[
            # MOP
            (0, 5),
            (0, 6),
            (3, 6),
            (1, 5),
            (4, 6),
            (2, 5),
            (5, 6),
            (6, 5),
            # PBMC
            # (0, 4),
            # (0, 6),
            # (1, 4),
            # (1, 6),
            # (2, 5),
            # (2, 6),
            # (3, 5),
            # (3, 6),
            # (4, 5),
            # (4, 6),
            # (5, 6),
            # (6, 5),
            # (6, 4),
            # (5, 4),
        ],  # 您的限制条件
    )

    # 开始训练
    result_dir = result_dir

    trainer.train(
        nodes=nodes,
        edges=sample_edges,
        feat_net=feat_net,
        feat_weight=feat_weight,
        num_weights=1,  # 根据需要调整
        num_epochs=num_epochs,
        val_split=val_split,
        edge_weights=sample_edge_weights,  # 使用边权重
        domain_labels=True,
        result_dir=result_dir,
    )

    # 绘制损失曲线
    trainer.plot_losses()


def evaluate_trained_embeddings(i,result_dir,mdata_path):

    result_dir = result_dir 
    neg_sampling_mode = "matched"
    add_feature_net = True
    neg_sample_in_batch = False
    # 读取数据
    mdata = mu.read(mdata_path)
    # import ipdb;ipdb.set_trace()
    node_df = pd.read_csv(NODES_PATH, index_col="idx")
    if add_feature_net:
        embeddings = pd.read_csv(
            os.path.join(
                result_dir, f"final_embeddings_{neg_sampling_mode}_add_feat.csv"
            ),
            index_col="node_index",
        )

    else:
        embeddings = pd.read_csv(
            os.path.join(result_dir, f"final_embeddings_{neg_sampling_mode}.csv"),
            index_col="node_index",
        )
    print(f"Embedding file loaded: {embeddings.shape}")
    print(f"Node file loaded: {node_df.shape}")

    # 对齐embedding index和node index
    mapped_embeddings = embeddings.copy()
    mapped_embeddings.index = node_df.loc[
        mapped_embeddings.index, "Unnamed: 0"
    ].values  # 替换索引
    aligned_embeddings = mapped_embeddings.reindex(
        mdata.obs.index
    )  # 确保 embedding 的索引和 mdata.obs 的索引对齐
    if aligned_embeddings.isnull().any().any():
        print(
            "Warning: Some embeddings could not be aligned with mdata.obs."
        )  # 检查对齐后的结果，如果有 NaN 行，表示有些嵌入无法对齐
        aligned_embeddings = aligned_embeddings.dropna()
    aligned_embeddings.to_csv(os.path.join(result_dir, "final_mapped_embeddings.csv"))
    print(
        f"Mapped embeddings saved to {os.path.join(result_dir, 'final_mapped_embeddings.csv')}"
    )

    # 将对齐后的 embeddings 加入 mdata
    mdata.obsm["X_embeddings"] = aligned_embeddings.values
    mdata.obs["cell_type"] = (
        mdata.obs["atac:cell_type"]
        .combine_first(mdata.obs["rna:cell_type"])
        .astype("category")
    )
    mdata.obs["batch"] = (
        mdata.obs["atac:batch"].combine_first(mdata.obs["rna:batch"]).astype("category")
    )
    # mdata.obs["rna:batch"] = mdata.obs["rna:batch"].astype("category")

    # 创建基于 'protein:coarse_cluster' 的 UMAP 可视化
    sc.pp.neighbors(mdata, use_rep="X_embeddings")  # 使用 embeddings 进行邻接矩阵计算
    sc.tl.umap(mdata)  # 计算 UMAP

    # 绘制 UMAP 图
    plt.figure(figsize=(8, 6))
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    axes = axes.flatten()  # 展开 axes 为一维数组，便于迭代
    sc.pl.umap(
        mdata,
        color=["batch"],
        ax=axes[0],
        title="UMAP of Batch Embeddings",
        show=True,
    )
    sc.pl.umap(
        mdata,
        color=["cell_type"],
        ax=axes[1],
        title="UMAP colored by protein",
        show=True,
    )
    plt.tight_layout()
    plt_path = os.path.join(result_dir, f"umap_plot_{neg_sampling_mode}.png")
    plt.savefig(plt_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"UMAP combined plot saved to '{plt_path}'")

    result_dir2 = "/home/yuyipei/graph_mosaic_integration/result/results"
    mdata_path = "/data/share_data/yuytest/gmi_data/MOP.h5mu"
    run_benchmark(
        mdata_path,
        result_dir,
        result_dir2,
    )


def plot_ablation():
    result_dir = f"/home/yuyipei/graph_mosaic_integration/result"
    # 初始化一个空的DataFrame，用于存储最终结果
    consolidated_df = pd.DataFrame()

    # 遍历result_path下的所有子文件夹
    for folder_name in os.listdir(result_dir):
        folder_path = os.path.join(result_dir, folder_name)

        # 检查是否为文件夹
        if os.path.isdir(folder_path):
            csv_file = os.path.join(folder_path, "benchmark_results.csv")

            # 检查benchmark_results.csv文件是否存在
            if os.path.exists(csv_file):
                # 读取CSV文件
                df = pd.read_csv(csv_file)

                # 假设第一列是指标名称，提取指标名称和'GMI_matched+batch_rem'列
                # 如果第一列有列名，可以使用具体的列名替代'Unnamed: 0'或类似名称
                # 这里假设第一列没有列名，使用iloc进行定位

                # 获取第一列（指标名称）
                metric_names = df.iloc[:, 0]
                # 获取'GMI_matched+batch_rem'列
                gmi_batch_rem = df["GMI_matched+batch_rem"]

                # 创建一个临时DataFrame
                temp_df = pd.DataFrame(
                    {"Metric": metric_names, folder_name: gmi_batch_rem}
                )

                # 如果是第一个文件夹，初始化consolidated_df
                if consolidated_df.empty:
                    consolidated_df = temp_df
                else:
                    # 合并DataFrame，基于'Metric'列
                    consolidated_df = pd.merge(
                        consolidated_df, temp_df, on="Metric", how="outer"
                    )
            else:
                print(f"警告: 文件 {csv_file} 不存在，跳过。")

    # 设置'Metric'列为第一列
    consolidated_df = consolidated_df[
        ["Metric"] + [col for col in consolidated_df.columns if col != "Metric"]
    ]

    # 保存合并后的DataFrame到新的CSV文件
    output_csv = os.path.join(result_dir, "consolidated_results.csv")
    consolidated_df.to_csv(output_csv, index=False)

    # 读取合并后的CSV文件
    df = pd.read_csv(output_csv)

    # 将数据从宽格式转换为长格式
    df_long = pd.melt(
        df, id_vars=["Metric"], var_name="Alpha", value_name="GMI_matched+batch_rem"
    )

    # 将Alpha列转换为数值类型（确保正确排序）
    df_long["Alpha"] = pd.to_numeric(df_long["Alpha"])

    # 排序Alpha值以确保折线图中x轴顺序正确
    df_long = df_long.sort_values("Alpha")

    # 设置绘图风格和调色板
    sns.set(style="whitegrid", palette="Set2")

    # 初始化绘图
    plt.figure(figsize=(14, 10))

    # 绘制折线图，使用不同的线型和标记
    line_plot = sns.lineplot(
        data=df_long,
        x="Alpha",
        y="GMI_matched+batch_rem",
        hue="Metric",
        marker="o",
        linewidth=2.5,
    )

    # 设置图表标题和标签
    plt.title("GMI_matched_batch_removal_smooth02", fontsize=18, weight="bold")
    plt.xlabel("Alpha(weight)", fontsize=16)
    plt.ylabel("value", fontsize=16)

    # 设置x轴为对数刻度
    plt.xscale("log")

    # 设置x轴刻度格式
    plt.xticks(sorted(df_long["Alpha"].unique()), rotation=45, fontsize=12)

    # 设置y轴刻度字体大小
    plt.yticks(fontsize=12)

    # 调整图例
    plt.legend(
        title="metric",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        fontsize=10,
        title_fontsize=12,
    )

    # 优化布局
    plt.tight_layout()

    # 保存图表为PNG格式
    output_plot = os.path.join(result_dir, "metrics_alpha_variation_log.png")
    plt.savefig(output_plot, dpi=300)
    print(f"折线图已保存到 {output_plot}")
