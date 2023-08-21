# %% imports and definition
import os

import numpy as np
import pandas as pd
import plotly.express as px
import xarray as xr
from sklearn.decomposition import PCA

from routine.utilities import load_mat_data

IN_DPATH = "./data"
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
    pca = PCA(n_components=PARAM_NCOMP, whiten=PARAM_WHITEN)
    proj = xr.DataArray(
        pca.fit_transform(act),
        dims=["frame", "comp"],
        coords={
            "frame": act.coords["frame"],
            "comp": ["comp{}".format(i) for i in range(PARAM_NCOMP)],
        },
    )
    proj_df = proj.to_pandas().merge(
        behav, on="frame", how="left", validate="one_to_one"
    )
    fig = px.scatter_3d(
        proj_df,
        x="comp0",
        y="comp1",
        z="comp2",
        symbol="target",
        color="event",
        symbol_map={"novel": "square", "familiar": "x", "self": "circle"},
    )
    fig.update_traces(marker_size=3)
    fig.update_layout(legend={"itemsizing": "constant"})
    fig.write_html(os.path.join(fig_path, "{}-{}.html".format(anm, ss)))
