# %% imports and definition
import os

import numpy as np
import pandas as pd
import plotly.express as px
import xarray as xr
from sklearn.decomposition import PCA

from routine.plotting import scatter_3d
from routine.utilities import load_mat_data

IN_DPATH = "./data"
IN_CELLMAP = "./data/CellMaps.xlsx"
PARAM_BEHAV_ORD = [
    "Familiar-BeingAttacked",
    "Novel-BeingAttacked",
    "BeingAttacked",
    "Familiar-SideAttack",
    "Novel-SideAttack",
    "SideAttack",
    "Familiar-AggressiveChase",
    "Novel-AggressiveChase",
    "AggressiveChase",
    "Familiar-Groupsniff_20fps",
    "Novel-Groupsniff_20fps",
    "Groupsniff_20fps",
    "Familiar-SideSniff1_20fps",
    "Novel-SideSniff1_20fps",
    "SideSniff1_20fps",
    "Familiar-dig_20fps",
    "Novel-dig_20fps",
    "dig_20fps",
    "Familiar-nose2nose1_20fpsr",
    "Novel-nose2nose1_20fpsr",
    "nose2nose1_20fpsr",
    "Familiar-nose2rear1_20fps",
    "Novel-nose2rear1_20fps",
    "nose2rear1_20fps",
    "Familiar-nosey1_20fps",
    "Novel-nosey1_20fps",
    "nosey1_20fps",
    "Familiar-follow1_20fps",
    "Novel-follow1_20fps",
    "follow1_20fps",
    "Familiar-groomfront_20fps",
    "Novel-groomfront_20fps",
    "groom_front",
    "groomfront_20fps",
    "sitcorner",
    "sitcorner_20fps",
    "wallclimb",
    "wallclimb_20fps",
    "walk",
    "walk_20fps",
    "still",
    "still_20fps",
]
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
FIG_PATH = "./figs/pca"


# %% pca analysis and plot projections
def behav_key(k):
    return list(map(PARAM_BEHAV_ORD.index, k))


def classify_behav(row):
    fm = int(row["frame"])
    row = row.drop("frame").sort_index(key=behav_key)
    if row.max() == 1:
        evt = row.idxmax()
        if evt.endswith("_20fps"):
            evt = evt[:-6]
        if evt.endswith("_20fpsr"):
            evt = evt[:-7]
        if evt.startswith("Novel-"):
            evt = evt.split("-")[1]
            tgt = "novel"
        elif evt.startswith("Familiar-"):
            evt = evt.split("-")[1]
            tgt = "familiar"
        else:
            tgt = "self"
    else:
        evt = np.nan
        tgt = np.nan
    return pd.Series({"frame": fm, "event": evt, "target": tgt})


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
