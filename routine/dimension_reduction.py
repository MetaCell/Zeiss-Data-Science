import numpy as np
import xarray as xr
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.preprocessing import StandardScaler


def reduce_wrap(
    algo, arr, n_components, samp_dim="frame", feat_dim="unit_id", whiten=True, **kwargs
):
    if whiten and algo != "pca":
        scalar = StandardScaler()
        arr = xr.apply_ufunc(
            scalar.fit_transform,
            arr,
            input_core_dims=[[samp_dim, feat_dim]],
            output_core_dims=[[samp_dim, feat_dim]],
        )
    if arr.sizes[feat_dim] > n_components:
        if algo == "pca":
            reduc = PCA(n_components=n_components, whiten=whiten, **kwargs)
        elif algo == "isomap":
            reduc = Isomap(n_components=n_components, **kwargs)
        else:
            raise NotImplementedError("Algorithm {} not implemented".format(algo))
        proj = xr.apply_ufunc(
            reduc.fit_transform,
            arr,
            input_core_dims=[[samp_dim, feat_dim]],
            output_core_dims=[[samp_dim, "comp"]],
        ).assign_coords({"comp": ["comp{}".format(i) for i in range(n_components)]})
        proj_df = proj.to_pandas()
    else:
        proj_df = arr.transpose(samp_dim, feat_dim).to_pandas()
        proj_df.columns = ["comp{}".format(i) for i in range(len(proj_df.columns))]
        for i in range(n_components):
            proj_df["comp{}".format(i)] = proj_df.get("comp{}".format(i), 0)
    return proj_df
