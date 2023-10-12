# %% imports and definition
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import xarray as xr
from pandas.api.types import union_categoricals
from scipy.ndimage import gaussian_filter1d

from routine.utilities import load_mat_data, parse_behav

IN_DPATH = "./data"
PARAM_BEHAV_WND = (-20 * 2.5, 20 * 2.5)
SS_DICT = {}
PARAM_NCOMP = 3
PARAM_NNB = 5
PARAM_WHITEN = False
PARAM_SIGMA = 4
FIG_PATH = "./figs/peth"
OUT_PATH = "./intermediate/peth"
ANMS = ["21271R", "21272R", "21965RL", "22384", "22670"]
SS = [
    "Acc1",
    "Acc2",
    "Acc3",
    "Acc4",
    "Acc5",
    "Train1",
    "Train2",
    "Train3",
    "Train4",
    "Test",
]
PARAM_CAT_COLS = [
    "animal",
    "session",
    "evt_id",
    "unit_id",
    "evt_fm",
    "evt",
    "by",
    "region",
]


def concat_cat(df_ls, cat_cols):
    dtypes = {c: union_categoricals([d[c] for d in df_ls]).dtype for c in cat_cols}
    df_ls = [d.astype(dtypes) for d in df_ls]
    return pd.concat(df_ls, ignore_index=True)


def agg_across(df, cols, val):
    agg_cols = list(set(df.columns) - set(cols) - set([val]))
    return df.groupby(agg_cols, observed=True)[val]


# %% load and aggregate events
for (anm, ss), act, curC, curS, behav_df in load_mat_data(
    IN_DPATH, return_behav="thresholded"
):
    ss_df = []
    act = xr.apply_ufunc(
        gaussian_filter1d,
        curS,
        input_core_dims=[["frame"]],
        output_core_dims=[["frame"]],
        vectorize=True,
        kwargs={"sigma": PARAM_SIGMA},
    )
    reg_dict = act.coords["region"].to_series().to_dict()
    behav_df = behav_df.set_index("frame")
    behav_parse = behav_df["behavior"].apply(parse_behav)
    behav = pd.concat([behav_df, behav_parse], axis="columns").reset_index()
    behav["evt"] = behav["event"] + "-" + behav["target"]
    for by in ["event", "event_raw"]:
        for evt, evt_df in behav.groupby(by):
            evt_arrs = []
            for fm in evt_df["frame"]:
                fm_dict = {
                    f: i
                    for f, i in zip(
                        np.arange(fm + PARAM_BEHAV_WND[0], fm + PARAM_BEHAV_WND[1] + 1),
                        np.arange(PARAM_BEHAV_WND[0], PARAM_BEHAV_WND[1] + 1),
                    )
                }
                a = act.sel(
                    frame=slice(fm + PARAM_BEHAV_WND[0], fm + PARAM_BEHAV_WND[1])
                ).assign_coords(evt_id="{}-{}".format(evt, fm))
                a = a.rename({"frame": "evt_fm"}).assign_coords(
                    evt_fm=a.coords["frame"].to_series().map(fm_dict).values.astype(int)
                )
                evt_arrs.append(a)
            edf = (
                xr.concat(evt_arrs, dim="evt_id")
                .rename("act")
                .to_series()
                .reset_index()
            )
            edf["evt"] = evt
            edf["by"] = by
            edf["region"] = edf["unit_id"].map(reg_dict)
            edf["animal"] = anm
            edf["session"] = ss
            edf["evt_id"] = edf["animal"] + "-" + edf["session"] + "-" + edf["evt_id"]
            edf = edf.astype({c: "category" for c in PARAM_CAT_COLS})
            ss_df.append(edf)
    ss_df = concat_cat(ss_df, PARAM_CAT_COLS)
    os.makedirs(OUT_PATH, exist_ok=True)
    ss_df.to_feather(os.path.join(OUT_PATH, "{}-{}.feat".format(anm, ss)))

# %% plot peth across animals
for ss in SS:
    ss_df = []
    for anm in ANMS:
        try:
            ss_dat = pd.read_feather(
                os.path.join(OUT_PATH, "{}-{}.feat".format(anm, ss))
            )
        except FileNotFoundError:
            continue
        ss_df.append(ss_dat[ss_dat["region"].notnull()])
    ss_df = concat_cat(ss_df, PARAM_CAT_COLS)
    for by, by_df in ss_df.groupby("by", observed=True):
        by_df["evt"] = by_df["evt"].cat.remove_unused_categories()
        g = sns.relplot(
            data=by_df,
            kind="line",
            x="evt_fm",
            y="act",
            hue="animal",
            row="region",
            col="evt",
            facet_kws={"margin_titles": True, "legend_out": True, "sharey": "row"},
            errorbar="se",
            err_style="band",
            height=2.5,
        )
        g.map(plt.axvline, x=0, color=".7", dashes=(2, 1), zorder=0)
        os.makedirs(FIG_PATH, exist_ok=True)
        fig = g.fig
        fig.savefig(os.path.join(FIG_PATH, "{}-by_{}.svg".format(ss, by)))
        plt.close(fig)

# %% plot peth for individual cell
for ss in SS:
    ss_df = []
    for anm in ANMS:
        try:
            ss_dat = pd.read_feather(
                os.path.join(OUT_PATH, "{}-{}.feat".format(anm, ss))
            )
        except FileNotFoundError:
            continue
        ss_dat = (
            agg_across(ss_dat[ss_dat["region"].notnull()], ["evt_id"], "act")
            .mean()
            .reset_index()
        )
        ss_dat["unit_id"] = (
            ss_dat["animal"].astype(str)
            + "-"
            + ss_dat["session"].astype(str)
            + "-"
            + ss_dat["unit_id"].astype(str)
        ).astype("category")
        ss_df.append(ss_dat)
    ss_df = concat_cat(ss_df, set(PARAM_CAT_COLS) - set(["evt_id"]))
    for by, by_df in ss_df.groupby("by", observed=True):
        by_df["evt"] = by_df["evt"].cat.remove_unused_categories()
        fig = px.line(
            by_df,
            x="evt_fm",
            y="act",
            line_group="unit_id",
            color="animal",
            facet_row="region",
            facet_col="evt",
        )
        os.makedirs(FIG_PATH, exist_ok=True)
        fig.write_html(os.path.join(FIG_PATH, "{}-by_{}-cells.html".format(ss, by)))
