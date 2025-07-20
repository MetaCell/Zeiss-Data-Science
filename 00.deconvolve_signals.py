# %% imports and definition
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import xarray as xr
from minian.cnmf import update_temporal_block

from routine.utilities import load_mat_data, norm_cells

IN_DPATH = "./data/20250711/"
OUT_PATH = "./intermediate/deconvolution"
FIG_PATH = "./figs/deconvolution"
PARAM_TEMP = {
    "p": 1,
    "noise_freq": 0.05,
    "add_lag": 20,
    "sparse_penal": 1,
    "max_iters": 200,
    "zero_thres": 1e-6,
    "med_wd": 500,
}
os.makedirs(OUT_PATH, exist_ok=True)
os.makedirs(FIG_PATH, exist_ok=True)


def plot_traces(trace_dict, fm_dim="frame", cell_dim="unit_id", gap=0.2, ncell=20):
    uids = np.array(list(trace_dict.values())[0].coords[cell_dim])
    ncell = min(ncell, len(uids))
    cells = np.random.choice(uids, ncell, replace=False)
    dfs = []
    for tr_name, a in trace_dict.items(): 
        a = a.sel({cell_dim: cells}).transpose(cell_dim, fm_dim)
        a_norm = norm_cells(a, fm_dim)
        a_norm = a_norm + (np.arange(ncell) * (1 + gap)).reshape((-1, 1))
        a_df = a_norm.rename("act").to_series().reset_index()
        a_df["signal"] = tr_name
        dfs.append(a_df)
    dfs = pd.concat(dfs, ignore_index=True)
    dfs["sig_id"] = dfs[cell_dim].astype(str) + "-" + dfs["signal"]
    return px.line(dfs, x=fm_dim, y="act", color="signal", line_group="sig_id")


# %% load data and deconvolve
for (anm, ss), mov,_, _, roi_ds, behav_df in load_mat_data(
    IN_DPATH, load_deconv=False, load_roi=True, load_act=True, return_behav="raw"
):
    roi = roi_ds['rois_raw'].dropna('height', how='all').dropna('width', how='all')
    act = roi.dot(mov)
    act = act.dropna("frame").transpose("unit_id", "frame")
    C, S, b, c0, g = update_temporal_block(np.array(act), **PARAM_TEMP)
    C = xr.DataArray(
        C,
        dims=["unit_id", "frame"],
        coords={
            "unit_id": np.array(act.coords["unit_id"]),
            "frame": np.array(act.coords["frame"]),
        },
    )
    S = xr.DataArray(
        S,
        dims=["unit_id", "frame"],
        coords={
            "unit_id": np.array(act.coords["unit_id"]),
            "frame": np.array(act.coords["frame"]),
        },
    )
    fig = plot_traces({"raw": act, "calcium": C, "deconv": S})
    fig.write_html(os.path.join(FIG_PATH, "{}-{}.html".format(anm, ss)))
    ds = xr.merge([act.rename('act'), C.rename("C"), S.rename("S")])
    ds.to_netcdf(os.path.join(OUT_PATH, "{}-{}.nc".format(anm, ss)))
