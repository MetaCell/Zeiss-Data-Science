# %% imports and definition
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import xarray as xr
from scipy.ndimage import gaussian_filter1d
from scipy.stats import zscore
from tqdm.auto import tqdm

from routine.utilities import agg_across, concat_cat, load_mat_data, parse_behav

IN_DPATH = "./data/20250711"
PARAM_BEHAV_WND = (-20 * 2.5, 20 * 2.5)
PARAM_SIGMA = 4
FIG_PATH = "./figs/peth"
OUT_PATH = "./intermediate/peth"
ANMS = ["YAS21271R", "YAS21272R", "YAS21965RL", "YAS22384", "YAS22670"]
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
SS_TYPE = {
    "Acc1": "Acc",
    "Acc2": "Acc",
    "Acc3": "Acc",
    "Acc4": "Acc",
    "Acc5": "Acc",
    "Train1": "Train",
    "Train2": "Train",
    "Train3": "Train",
    "Train4": "Train",
    "Test": "Test",
}
PARAM_BEHAV_CUR = {
    "Acc": {
        "Active": ["dig-self", "walk-self", "wallclimb-self", "groom_front-self"],
        "NonActive": ["still-self", "sitcorner-self"],
    },
    "Train": {
        "Target of Aggression": ["BeingAttacked-self"],
        "Aggressor": ["SideAttack-self", "AggressiveChase-self", "nosey1-self"],
        "Social": ["nose2nose1-self", "nose2rear1-self", "follow1-self"],
        "NonSocial": ["dig-self", "still-self", "sitcorner-self"],
    },
    "Test": {
        "Target of Aggression": ["BeingAttacked-novel", "BeingAttacked-familiar"],
        "Aggressor": [
            "SideAttack-familiar",
            "SideAttack-novel",
            "AggressiveChase-familiar",
            "AggressiveChase-novel",
            "nosey1-familiar",
            "nosey1-novel",
        ],
        "Aggressor: Familiar": [
            "SideAttack-familiar",
            "AggressiveChase-familiar",
            "nosey1-familiar",
        ],
        "Aggressor: Novel": [
            "SideAttack-novel",
            "AggressiveChase-novel",
            "nosey1-novel",
        ],
        "Social: General": [
            "nose2nose1-familiar",
            "nose2nose1-novel",
            "nose2rear1-familiar",
            "nose2rear1-novel",
            "follow1-familiar",
            "follow1-novel",
            "SideSniff1-familiar",
            "SideSniff1-novel",
            "Groupsniff-familiar",
            "Groupsniff-novel",
        ],
        "Social: Familiar": [
            "nose2nose1-familiar",
            "nose2rear1-familiar",
            "follow1-familiar",
            "SideSniff1-familiar",
        ],
        "Social: Novel": [
            "nose2nose1-novel",
            "nose2rear1-novel",
            "follow1-novel",
            "SideSniff1-novel",
        ],
        "NonSocial": ["dig-self", "still-self", "sitcorner-self", "groom_front-self"],
        "Active": ["dig-self", "walk-self", "wallclimb-self", "groom_front-self"],
        "NonActive": ["still-self", "sitcorner-self"],
    },
}


# %% load and aggregate events
for (anm, ss), act, curC, curS, cur_roi, behav_df in load_mat_data(
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
    print(act.max())
    reg_dict = act.coords["region"].to_series().to_dict()
    behav_df = behav_df.set_index("frame")
    behav_parse = behav_df["behavior"].apply(parse_behav)
    behav = pd.concat([behav_df, behav_parse], axis="columns").reset_index()
    behav["evt"] = behav["event"] + "-" + behav["target"]
    # build new curated behavior
    behav["evt_raw-tgt"] = behav["event_raw"] + "-" + behav["target"]
    ss_type = SS_TYPE[ss]
    behav_cur = []
    for beh_cur, beh_raw in PARAM_BEHAV_CUR[ss_type].items():
        cur_beh = behav[behav["evt_raw-tgt"].isin(beh_raw)].copy()
        cur_beh["event_cur"] = beh_cur
        behav_cur.append(cur_beh)
    behav_cur = pd.concat(behav_cur, ignore_index=True)
    beh_covered = set(sum(PARAM_BEHAV_CUR[ss_type].values(), []))
    beh_miss = set(behav["evt_raw-tgt"].dropna().unique()) - beh_covered
    if beh_miss:
        warnings.warn(
            "Following behavior was not curated in animal {} session {}: {}".format(
                anm, ss, beh_miss
            )
        )
    behav_dfs = {"event": behav, "event_raw": behav, "event_cur": behav_cur}
    for by, bdf in behav_dfs.items():
        for evt, evt_df in bdf.groupby(by):
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

# %% plot peth
for cell_norm in ["raw", "zscore"]:
    for ss in tqdm(SS):
        ss_df = []
        for anm in ANMS:
            try:
                ss_dat = pd.read_feather(
                    os.path.join(OUT_PATH, "{}-{}.feat".format(anm, ss))
                )
            except FileNotFoundError:
                continue
            ss_dat = ss_dat[ss_dat["region"].notnull()]
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
            if cell_norm == "zscore":
                ss_dat["act"] = agg_across(ss_dat, ["evt_fm"], "act").transform(zscore)
                ss_dat = ss_dat.dropna(subset="act")
            ss_df.append(ss_dat)
        ss_df = concat_cat(ss_df, set(PARAM_CAT_COLS) - set(["evt_id"]))
        for by, by_df in ss_df.groupby("by", observed=True):
            by_df["evt"] = by_df["evt"].cat.remove_unused_categories()
            # summary plot
            fig_path = os.path.join(FIG_PATH, "summary-{}".format(cell_norm))
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
            os.makedirs(fig_path, exist_ok=True)
            fig = g.fig
            fig.savefig(os.path.join(fig_path, "{}-by_{}.svg".format(ss, by)))
            plt.close(fig)
            # line plot
            fig_path = os.path.join(FIG_PATH, "individual-{}".format(cell_norm))
            fig = px.line(
                by_df,
                x="evt_fm",
                y="act",
                line_group="unit_id",
                color="animal",
                facet_row="region",
                facet_col="evt",
            )
            os.makedirs(fig_path, exist_ok=True)
            fig.write_html(os.path.join(fig_path, "{}-by_{}.html".format(ss, by)))
