# %% imports and definition
import os

import numpy as np
import pandas as pd
import plotly.express as px
import xarray as xr
from scipy.ndimage import gaussian_filter1d

from routine.dimension_reduction import reduce_wrap
from routine.utilities import classify_behav, load_mat_data

IN_DPATH = "./data"
IN_CELLMAP = "./data/CellMaps.xlsx"
PARAM_BEHAV_WND = (-20 * 2.5, 20 * 2.5)
SS_DICT = {}
PARAM_NCOMP = 3
PARAM_NNB = 5
PARAM_WHITEN = False
PARAM_SIGMA = 4
FIG_PATH = "./figs/cell_reduc"
OUT_PATH = "./intermediate/cell_reduc"

# %% load and aggregate events
tuning_ls = []
for (anm, ss), act, curC, curS, behav_df in load_mat_data(IN_DPATH):
    act = xr.apply_ufunc(
        gaussian_filter1d,
        curS,
        input_core_dims=[["frame"]],
        output_core_dims=[["frame"]],
        vectorize=True,
        kwargs={"sigma": PARAM_SIGMA},
    )
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
    reg_dict = evt_tuning.coords["region"].to_series().to_dict()
    tuning_df = evt_tuning.rename("act").to_series().reset_index()
    tuning_df["region"] = tuning_df["unit_id"].map(reg_dict)
    tuning_df["animal"] = anm
    tuning_df["session"] = ss
    tuning_ls.append(tuning_df)
tuning_df = pd.concat(tuning_ls, ignore_index=True).astype(
    {"event": "category", "animal": "category", "session": "category"}
)
os.makedirs(OUT_PATH, exist_ok=True)
tuning_df.to_feather(os.path.join(OUT_PATH, "tuning_df.feat"))

# %% plot per session/animal projections
tuning_df = pd.read_feather(os.path.join(OUT_PATH, "tuning_df.feat"))
for trim in [True, False]:
    for algo, params in {"pca": dict(), "isomap": {"n_neighbors": PARAM_NNB}}.items():
        fig_path = os.path.join(FIG_PATH, algo)
        os.makedirs(fig_path, exist_ok=True)
        for (anm, ss), tdf in tuning_df.groupby(["animal", "session"], observed=True):
            evt_tuning = tdf.set_index(["unit_id", "event"])["act"].to_xarray()
            proj_df = reduce_wrap(
                algo,
                evt_tuning,
                PARAM_NCOMP,
                samp_dim="unit_id",
                feat_dim="event",
                whiten=PARAM_WHITEN,
                **params
            )
            proj_df = proj_df.reset_index()
            proj_df["region"] = proj_df["unit_id"].map(
                tdf.set_index("unit_id")["region"].to_dict()
            )
            if trim:
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
            fig.write_html(
                os.path.join(
                    fig_path, "{}-{}-{}.html".format(anm, ss, "trim" if trim else "all")
                )
            )

# %% plot across animals
tuning_df = pd.read_feather(os.path.join(OUT_PATH, "tuning_df.feat"))
for trim in [True, False]:
    for algo, params in {"pca": dict(), "isomap": {"n_neighbors": PARAM_NNB}}.items():
        fig_path = os.path.join(FIG_PATH, algo)
        os.makedirs(fig_path, exist_ok=True)
        for ss, tdf in tuning_df.groupby("session"):
            tdf["uid"] = tdf["animal"].astype(str) + "-" + tdf["unit_id"].astype(str)
            evt_tuning = tdf.set_index(["uid", "event"])["act"].to_xarray().fillna(0)
            proj_df = reduce_wrap(
                algo,
                evt_tuning,
                PARAM_NCOMP,
                samp_dim="uid",
                feat_dim="event",
                whiten=PARAM_WHITEN,
                **params
            )
            proj_df = proj_df.reset_index()
            proj_df["region"] = proj_df["uid"].map(
                tdf.set_index("uid")["region"].to_dict()
            )
            if trim:
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
            fig.write_html(
                os.path.join(
                    fig_path, "all-{}-{}.html".format(ss, "trim" if trim else "all")
                )
            )
