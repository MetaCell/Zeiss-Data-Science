# %% imports and definition
import os

import numpy as np
import pandas as pd
import plotly.express as px
import xarray as xr
from sklearn.decomposition import PCA

from routine.plotting import scatter_3d
from routine.utilities import classify_behav, load_mat_data

IN_DPATH = "./data"
IN_CELLMAP = "./data/CellMaps.xlsx"

PARAM_COLMAP = {
    "BeingAttacked": "lightcoral",
    "SideAttack": "crimson",
    "AggressiveChase": "darkred",
    "Groupsniff": "violet",
    "SideSniff1": "darkviolet",
    "dig": "springgreen",
    "nose2nose1": "purple",
    "nose2rear1": "magenta",
    "nosey1": "mediumorchid",
    "follow1": "deepskyblue",
    "groom_front": "green",
    "groomfront": "green",
    "sitcorner": "mediumaquamarine",
    "wallclimb": "silver",
    "walk": "slategray",
    "still": "gray",
}
PARAM_SYMMAP = {"novel": "square", "familiar": "x", "self": "circle"}
SS_DICT = {}
PARAM_NCOMP = 3
PARAM_WHITEN = True
FIG_PATH = "./figs/frame_pca"


# %% pca analysis and plot projections
fig_path = os.path.join(FIG_PATH, "proj")
os.makedirs(fig_path, exist_ok=True)
for (anm, ss), act, behav_df in load_mat_data(IN_DPATH):
    behav = behav_df.apply(classify_behav, axis="columns")
    act = act.assign_coords(event=("frame", behav["event"]))
    act = act.dropna("frame").transpose("frame", "unit_id")
    act = act.sel(frame=act.coords["event"].notnull())
    projs = []
    for reg, reg_act in act.groupby("region"):
        if reg_act.sizes["unit_id"] > PARAM_NCOMP:
            pca = PCA(n_components=PARAM_NCOMP, whiten=PARAM_WHITEN)
            proj = xr.DataArray(
                pca.fit_transform(reg_act),
                dims=["frame", "comp"],
                coords={
                    "frame": act.coords["frame"],
                    "comp": ["comp{}".format(i) for i in range(PARAM_NCOMP)],
                },
            )
            proj_df = proj.to_pandas()
        else:
            proj_df = reg_act.to_pandas()
            proj_df.columns = ["comp{}".format(i) for i in range(len(proj_df.columns))]
        proj_df = proj_df.merge(behav, on="frame", how="left", validate="one_to_one")
        proj_df["region"] = reg + "-dim:{}".format(reg_act.sizes["unit_id"])
        projs.append(proj_df)
    proj_df = pd.concat(projs, ignore_index=True)
    proj_df = proj_df[proj_df["event"].notnull()].copy()
    proj_df["color"] = proj_df["event"].map(PARAM_COLMAP)
    proj_df["symbol"] = proj_df["target"].map(PARAM_SYMMAP)
    proj_df["legend"] = proj_df["target"] + "-" + proj_df["event"]
    proj_df[["comp0", "comp1", "comp2"]] = proj_df[["comp0", "comp1", "comp2"]].fillna(
        0
    )
    fig = scatter_3d(
        proj_df,
        facet_row=None,
        facet_col="region",
        col_wrap=3,
        x="comp0",
        y="comp1",
        z="comp2",
        legend_dim="legend",
        marker={"color": "color", "symbol": "symbol"},
        mode="markers",
    )
    fig.update_traces(marker_size=2)
    fig.update_layout(legend={"itemsizing": "constant"})
    fig.write_html(os.path.join(fig_path, "{}-{}.html".format(anm, ss)))
