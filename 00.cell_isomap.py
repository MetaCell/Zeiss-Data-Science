# %% imports and definition
import os

import numpy as np
import pandas as pd
import plotly.express as px
import xarray as xr
from sklearn.manifold import Isomap
from sklearn.preprocessing import StandardScaler

from routine.utilities import classify_behav, load_mat_data

IN_DPATH = "./data"
IN_CELLMAP = "./data/CellMaps.xlsx"
PARAM_BEHAV_WND = (-20 * 2.5, 20 * 2.5)
SS_DICT = {}
PARAM_NNB = 5
PARAM_NCOMP = 3
PARAM_WHITEN = True
PARAM_TRIM_UNKNOWN = True
FIG_PATH = "./figs/cell_isomap"

# %% pca analysis and plot projections
fig_path = os.path.join(FIG_PATH, "proj")
os.makedirs(fig_path, exist_ok=True)
for (anm, ss), act, behav_df in load_mat_data(IN_DPATH):
    behav = behav_df.apply(classify_behav, axis="columns")
    behav["evt"] = behav["event_map"] + "-" + behav["target"]
    evt_tuning = []
    for evt, evt_df in behav.groupby("evt"):
        evt_arrs = []
        for fm in evt_df["frame"]:
            a = act.sel(
                frame=slice(fm + PARAM_BEHAV_WND[0], fm + PARAM_BEHAV_WND[1])
            ).mean("frame")
            evt_arrs.append(a)
        evt_arr = xr.concat(evt_arrs, dim="evt").mean("evt").assign_coords(event=evt)
        evt_tuning.append(evt_arr)
    evt_tuning = xr.concat(evt_tuning, "event")
    if PARAM_WHITEN:
        scalar = StandardScaler()
        evt_tuning = xr.apply_ufunc(
            scalar.fit_transform,
            evt_tuning,
            input_core_dims=[["unit_id", "event"]],
            output_core_dims=[["unit_id", "event"]],
        )
    if evt_tuning.sizes["event"] > PARAM_NCOMP:
        iso = Isomap(n_components=PARAM_NCOMP, n_neighbors=PARAM_NNB)
        proj = xr.DataArray(
            iso.fit_transform(evt_tuning.transpose("unit_id", "event")),
            dims=["unit_id", "comp"],
            coords={
                "unit_id": evt_tuning.coords["unit_id"],
                "comp": ["comp{}".format(i) for i in range(PARAM_NCOMP)],
            },
        )
        proj_df = proj.to_pandas()
    else:
        proj_df = evt_tuning.transpose("unit_id", "event").to_pandas()
        proj_df.columns = ["comp{}".format(i) for i in range(len(proj_df.columns))]
        proj_df["comp0"] = proj_df.get("comp0", 0)
        proj_df["comp1"] = proj_df.get("comp1", 0)
        proj_df["comp2"] = proj_df.get("comp2", 0)
    proj_df = proj_df.reset_index()
    proj_df["region"] = proj_df["unit_id"].map(
        evt_tuning.coords["region"].to_series().to_dict()
    )
    if PARAM_TRIM_UNKNOWN:
        proj_df = proj_df.dropna()
    else:
        proj_df["region"] = proj_df["region"].fillna("unknown")
    fig = px.scatter_3d(
        proj_df,
        x="comp0",
        y="comp1",
        z="comp2",
        color="region",
        color_discrete_map={"unknown": "grey"},
    )
    fig.update_traces(marker_size=5)
    fig.update_layout(legend={"itemsizing": "constant"})
    fig.write_html(os.path.join(fig_path, "{}-{}.html".format(anm, ss)))
