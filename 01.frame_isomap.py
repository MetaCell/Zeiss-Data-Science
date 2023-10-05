# %% imports and definition
import itertools as itt
import os

import numpy as np
import pandas as pd
import plotly.express as px
import xarray as xr
from plotly.express.colors import qualitative
from sklearn.manifold import Isomap
from sklearn.preprocessing import StandardScaler

from routine.plotting import scatter_3d
from routine.utilities import classify_behav, load_mat_data

IN_DPATH = "./data"
IN_CELLMAP = "./data/CellMaps.xlsx"
PARAM_COLS = qualitative.Plotly
PARAM_COLMAP = {
    "attack": PARAM_COLS[1],
    "aggression": PARAM_COLS[4],
    "sniff": PARAM_COLS[2],
    "social": PARAM_COLS[0],
    "groom": PARAM_COLS[5],
    "move": PARAM_COLS[3],
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
PARAM_SZMAP = {"novel": 2.5, "familiar": 1.1, "self": 2}
PARAM_NCOMP = 3
PARAM_NNB = 25
FIG_PATH = "./figs/frame_isomap"
OUT_PATH = "./intermediate/frame_isomap"

os.makedirs(FIG_PATH, exist_ok=True)
os.makedirs(OUT_PATH, exist_ok=True)


def run_isomap(act, whiten=True):
    if whiten:
        scalar = StandardScaler()
        act = xr.apply_ufunc(
            scalar.fit_transform,
            act,
            input_core_dims=[["frame", "unit_id"]],
            output_core_dims=[["frame", "unit_id"]],
        )
    if act.sizes["unit_id"] > PARAM_NCOMP:
        iso = Isomap(n_neighbors=PARAM_NNB, n_components=PARAM_NCOMP, n_jobs=-1)
        proj = xr.DataArray(
            iso.fit_transform(act),
            dims=["frame", "comp"],
            coords={
                "frame": act.coords["frame"],
                "comp": ["comp{}".format(i) for i in range(PARAM_NCOMP)],
            },
        )
        proj_df = proj.to_pandas()
    else:
        proj_df = act.to_pandas()
        proj_df.columns = ["comp{}".format(i) for i in range(len(proj_df.columns))]
    proj_df = proj_df.merge(behav, on="frame", how="left", validate="one_to_one")
    return proj_df


# %% isomap analysis
fig_path = os.path.join(FIG_PATH, "proj")
os.makedirs(fig_path, exist_ok=True)
for (anm, ss), act, behav_df in load_mat_data(IN_DPATH):
    behav = behav_df.apply(classify_behav, axis="columns")
    act = act.assign_coords(event=("frame", behav["event"]))
    act = act.dropna("frame").transpose("frame", "unit_id")
    proj_ls = []
    for pts, whit in itt.product(["full", "events"], [True, False]):
        if pts == "events":
            cur_act = act.sel(frame=act.coords["event"].notnull())
        elif pts == "full":
            cur_act = act
        else:
            continue
        proj_df = run_isomap(cur_act, whit)
        proj_df["region"] = "all-dim:{}".format(cur_act.sizes["unit_id"])
        projs = [proj_df]
        for reg, reg_act in cur_act.groupby("region"):
            proj_df = run_isomap(reg_act)
            proj_df["region"] = reg + "-dim:{}".format(reg_act.sizes["unit_id"])
            projs.append(proj_df)
        proj_df = pd.concat(projs, ignore_index=True)
        proj_df["pts"] = pts
        proj_df["whiten"] = whit
        proj_df["animal"] = anm
        proj_df["session"] = ss
        proj_df = proj_df
        proj_ls.append(proj_df)
    proj_df = pd.concat(proj_ls).astype(
        {
            "pts": "category",
            "animal": "category",
            "session": "category",
            "event": "category",
            "target": "category",
            "event_map": "category",
            "region": "category",
        }
    )
    proj_df.to_feather(os.path.join(OUT_PATH, "{}-{}.feat".format(anm, ss)))


# %% plot projections
for proj_file in os.listdir(OUT_PATH):
    ss_df = pd.read_feather(os.path.join(OUT_PATH, proj_file))
    anm, ss = os.path.splitext(proj_file)[0].split("-")
    for (pts, whit), proj_df in ss_df.groupby(["pts", "whiten"], observed=True):
        fig_path = os.path.join(
            FIG_PATH, "proj-{}-{}".format(pts, "whiten" if whit else "raw")
        )
        os.makedirs(fig_path, exist_ok=True)
        proj_df = proj_df[proj_df["event"].notnull()].copy()
        for by in ["event", "frame"]:
            if by == "event":
                proj_df["color"] = proj_df["event_map"].map(PARAM_COLMAP)
                proj_df["symbol"] = proj_df["target"].map(PARAM_SYMMAP)
                proj_df["size"] = proj_df["target"].map(PARAM_SZMAP)
                mk_args = {"color": "color", "symbol": "symbol", "size": "size"}
            elif by == "frame":
                proj_df["size"] = 2
                mk_args = {"color": "frame", "size": "size"}
            proj_df["legend"] = (
                proj_df["target"].astype(str) + "-" + proj_df["event"].astype(str)
            )
            proj_df[["comp0", "comp1", "comp2"]] = proj_df[
                ["comp0", "comp1", "comp2"]
            ].fillna(0)
            fig = scatter_3d(
                proj_df,
                facet_row=None,
                facet_col="region",
                col_wrap=3,
                x="comp0",
                y="comp1",
                z="comp2",
                legend_dim="legend",
                marker=mk_args,
                mode="markers",
            )
            fig.update_layout(legend={"itemsizing": "constant"})
            fig.write_html(
                os.path.join(fig_path, "{}-{}-by_{}.html".format(anm, ss, by))
            )
