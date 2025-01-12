import os
import os.path as osp
from time import perf_counter
from argparse import ArgumentParser
from collections import defaultdict
import numpy as np
import scipy.sparse as sp
import anndata as ad
import mudata as md
import scanpy as sc
import scmomat
import sys
from gmi import (
    data_infor_integrate,
)
sys.path.append(os.path.abspath("src/gmi/mmAAVI"))

from preprocess import merge_obs_from_all_modalities

def main():
    parser = ArgumentParser()
    parser.add_argument("--preproc_data_dir", default="/data/share_data/yuytest/gmi_data/")
    parser.add_argument("--preproc_data_name", default="muto")  # 更新为 muto 数据集
    parser.add_argument("--results_dir", default="/home/yuyipei/graph_mosaic_integration/result")
    parser.add_argument("--results_name", default="muto_comparison")  # 更新为 muto 数据集的结果名
    parser.add_argument("--not_use_pseudo", action="store_true")
    parser.add_argument("--seeds", default=list(range(6)), type=int, nargs="+")
    parser.add_argument("--scmomat_device", default="cuda:0")
    parser.add_argument(
        "--methods", default=("scmomat"), nargs="+"
    )
    args = parser.parse_args()

    # ========================================================================
    # load preprocessed data
    # ========================================================================
    print("-- load preprocessed data --")
    mdata_fn = osp.join(
        args.preproc_data_dir, f"{args.preproc_data_name}.h5mu"
    )
    os.makedirs(args.results_dir, exist_ok=True)
    mdata = md.read(mdata_fn)
    mdata.mod['atac'].obs['batch'] = mdata.mod['atac'].obs['batch'].cat.codes+1
    mdata.mod['rna'].obs['batch'] = mdata.mod['rna'].obs['batch'].cat.codes+6
    merge_obs_from_all_modalities(mdata, key="cell_type")
    mdata = data_infor_integrate(
        mdata,
        feature_key="batch",
        saved_feature_name="batch",
        target_attr="obs",
    )
    mdata.obs['batch'] = mdata.obs['batch'].cat.codes+1
    print(mdata)
    print(mdata.obs['batch'].value_counts())

    # prepare the container to hold the results
    res_adata = ad.AnnData(obs={"placeholder": np.arange(mdata.n_obs)})

    # ========================================================================
    # rearrange the data
    # ========================================================================
    print("-- rearrange data --")
    batch_name = "batch"  # 根据数据结构，你可以修改这个为"batch5"等
    
    batch_uni = mdata.obs[batch_name].unique()
    batch_uni.sort()
    nbatches = batch_uni.shape[0]

    counts = {}
    for k, adat in mdata.mod.items():
        batch_uni_k = adat.obs[batch_name].unique()
        counts_k = []
        for bi in batch_uni:
            if bi in batch_uni_k:
                counts_k.append(adat.X[adat.obs[batch_name] == bi, :])
            else:
                counts_k.append(None)
        counts[k] = counts_k

    if not args.not_use_pseudo:
        net = mdata.varp["net"]
        atac_rna = net[mdata.varm["atac"], :][:, mdata.varm["rna"]].toarray()

        for i, arr_rna in enumerate(counts["rna"]):
            if arr_rna is None:
                arr_atac = counts["atac"][i]
                if arr_atac is not None:
                    counts["rna"][i] = ((arr_atac @ atac_rna) != 0).astype(int)

    # ========================================================================
    # running scmomat
    # ========================================================================
    if "scmomat" in args.methods:
        print("-- scmomat: preprocessing --")
        counts_scmomat = defaultdict(list)
        for k, arrs in counts.items():
            for dati in arrs:
                if dati is None:
                    counts_scmomat[k].append(None)
                    continue

                if sp.issparse(dati):
                    dati = dati.toarray()
                if k == "atac":
                    dati = scmomat.preprocess(dati, modality="ATAC")
                else:
                    dati = scmomat.preprocess(
                        dati, modality="RNA", log=True
                    )
                counts_scmomat[k].append(dati)

        counts_scmomat["nbatches"] = nbatches
        counts_scmomat["feats_name"] = {
            k: adati.var.index.values for k, adati in mdata.mod.items()
        }

        print("-- scmomat: modeling --")
        res_timing = []
        for seedi in args.seeds:
            print(f"-- scmomat: modeling, seed is {seedi} --")
            start_time = perf_counter()
            model = scmomat.scmomat_model(
                counts=counts_scmomat,
                K=30,
                batch_size=0.1,
                interval=1000,
                lr=1e-2,
                lamb=0.001,
                seed=seedi,
                device=args.scmomat_device,
            )
            model.train_func(T=4000)
            end_time = perf_counter()
            res_timing.append((seedi, end_time - start_time))

            zs = model.extract_cell_factors()
            res_adata.obsm[f"scMoMaT_s{seedi}"] = np.concatenate(zs)

        res_adata.uns["timing"] = {"scMoMaT": res_timing}

    # ========================================================================
    # save the results
    # ========================================================================
    print("-- save the results --")
    res_adata.write(osp.join(args.results_dir, f"{args.results_name}.h5ad"))

if __name__ == "__main__":
    main()
