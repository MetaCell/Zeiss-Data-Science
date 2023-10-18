# %% imports and definition
import itertools as itt
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import statsmodels.api as sm
import xarray as xr
from scipy.stats import zscore
from statsmodels.formula.api import ols
from tqdm.auto import tqdm

from routine.responsive_cells import compute_dff
from routine.utilities import agg_across, concat_cat

IN_DPATH = "./data"
PARAM_BEHAV_WND = (-20 * 2.5, 20 * 2.5)
SS_DICT = {}
PARAM_NCOMP = 3
PARAM_NNB = 5
PARAM_WHITEN = False
PARAM_SIGMA = 4

IN_PETH_PATH = "./intermediate/peth"
FIG_PATH = "./figs/responsive_cells"
OUT_PATH = "./intermediate/responsive_cells"
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


def response_anova(df, by="evt", y="dff"):
    by_var = "C({})".format(by)
    model = ols("{} ~ {}".format(y, by_var), data=df).fit()
    anova = sm.stats.anova_lm(model, typ=2)
    return anova.loc[by_var].rename(None)


# %% compute dffs
dffs = []
os.makedirs(OUT_PATH, exist_ok=True)
for datfile in tqdm(os.listdir(IN_PETH_PATH)):
    ss_dat = pd.read_feather(os.path.join(IN_PETH_PATH, datfile))
    ss_dat = ss_dat[ss_dat["region"].notnull()]
    dff = (
        agg_across(ss_dat, ["evt_fm"], "act", return_val=False)
        .apply(compute_dff)
        .rename("dff")
        .reset_index()
    )
    dffs.append(dff)
dffs = pd.concat(dffs, ignore_index=True)
dffs.to_feather(os.path.join(OUT_PATH, "dff.feat"))

# %% run tests
dffs = pd.read_feather(os.path.join(OUT_PATH, "dff.feat"))
sigs = (
    agg_across(dffs, ["evt_id", "evt"], "dff", return_val=False)
    .apply(response_anova)
    .reset_index()
)
sigs.to_feather(os.path.join(OUT_PATH, "sigs.feat"))

# %% plot peth of significant cells
sigs = pd.read_feather(os.path.join(OUT_PATH, "sigs.feat"))
sig_cells = sigs[sigs["PR(>F)"] < 0.01].copy()
sig_cells["unit_id"] = (
    sig_cells["by"].astype(str)
    + "-"
    + sig_cells["animal"].astype(str)
    + "-"
    + sig_cells["session"].astype(str)
    + "-"
    + sig_cells["unit_id"].astype(str)
)
for cell_norm in ["raw", "zscore"]:
    for ss in tqdm(SS):
        ss_df = []
        for anm in ANMS:
            try:
                ss_dat = pd.read_feather(
                    os.path.join(IN_PETH_PATH, "{}-{}.feat".format(anm, ss))
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
                ss_dat["by"].astype(str)
                + "-"
                + ss_dat["animal"].astype(str)
                + "-"
                + ss_dat["session"].astype(str)
                + "-"
                + ss_dat["unit_id"].astype(str)
            ).astype("category")
            ss_dat = ss_dat[ss_dat["unit_id"].isin(sig_cells["unit_id"])]
            if cell_norm == "zscore":
                ss_dat["act"] = agg_across(
                    ss_dat, ["unit_id", "evt_fm"], "act"
                ).transform(zscore)
                ss_dat = ss_dat.dropna(subset="act")
            ss_df.append(ss_dat)
        ss_df = concat_cat(ss_df, set(PARAM_CAT_COLS) - set(["evt_id"]))
        for by, by_df in ss_df.groupby("by", observed=True):
            by_df["evt"] = by_df["evt"].cat.remove_unused_categories()
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
