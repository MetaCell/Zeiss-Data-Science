# %% imports and definition
import itertools as itt
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import statsmodels.api as sm
import xarray as xr
from plotly.express.colors import qualitative
from scipy.stats import ttest_rel, zscore
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import multipletests
from tqdm.auto import tqdm

from routine.plotting import add_color_opacity, map_colors
from routine.responsive_cells import compute_dff
from routine.utilities import agg_across, concat_cat, load_mat_data

IN_DPATH = "./data"
PARAM_BEHAV_WND = (-20 * 2.5, 20 * 2.5)
SS_DICT = {}
PARAM_NCOMP = 3
PARAM_NNB = 5
PARAM_WHITEN = False
PARAM_SIGMA = 4

IN_PETH_PATH = "./intermediate/peth"
FIG_PATH = "./figs/specific_event_responses"
OUT_PATH = "./intermediate/specific_event_responses"
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


def response_ttest(df, pre_col="pre-evt", post_col="post-evt"):
    tres = ttest_rel(df[post_col], df[pre_col])
    return pd.Series({"ts": tres.statistic, "pval": tres.pvalue, "dof": tres.df})


def tt_correction(df, pval_col="pval", alpha=0.05, method="bonferroni"):
    rej, pvals_corr, _, alphacBonf = multipletests(df[pval_col], alpha, method=method)
    df["rej"] = rej
    df["pval_corr"] = pvals_corr
    df["alphacBonf"] = alphacBonf
    df["rej_any"] = df["rej"].any()
    return df


# %% compute mean responses
resp_df = []
os.makedirs(OUT_PATH, exist_ok=True)
for datfile in tqdm(os.listdir(IN_PETH_PATH)):
    ss_dat = pd.read_feather(os.path.join(IN_PETH_PATH, datfile))
    ss_dat = ss_dat[ss_dat["region"].notnull()]
    resp = (
        agg_across(ss_dat, ["evt_fm"], "act", return_val=False)
        .apply(compute_dff, return_segments=True)
        .reset_index()
    )
    resp_df.append(resp)
resp_df = pd.concat(resp_df, ignore_index=True)
resp_df.to_feather(os.path.join(OUT_PATH, "resp_df.feat"))

# %% run tests
resp_df = pd.read_feather(os.path.join(OUT_PATH, "resp_df.feat"))
resp_tt = (
    agg_across(
        resp_df,
        ["evt_id"],
        ["pre-evt", "post-evt"],
        multiple_val=True,
        return_val=False,
    )
    .apply(response_ttest)
    .reset_index()
)
resp_tt = (
    agg_across(
        resp_tt, ["evt"], ["pval", "ts", "dof"], multiple_val=True, return_val=False
    )
    .apply(tt_correction)
    .reset_index(drop=True)
)
resp_tt.to_feather(os.path.join(OUT_PATH, "resp_tt.feat"))
resp_tt.to_csv(os.path.join(OUT_PATH, "resp_tt.csv"), index=False)

# %% build sankey diagram
fig_path = FIG_PATH
os.makedirs(fig_path, exist_ok=True)
resp_tt = pd.read_feather(os.path.join(OUT_PATH, "resp_tt.feat"))
for (ss, by), ss_df in resp_tt.groupby(["session", "by"], observed=True):
    ss_df["evt-sign"] = (
        ss_df["evt"]
        + "-"
        + np.sign(ss_df["ts"]).map({1: "activated", -1: "suppressed"})
    )
    ss_df["unit_id"] = ss_df["animal"] + "-" + ss_df["unit_id"].astype(str)
    sig_df = ss_df[ss_df["rej_any"]].copy()
    # build nodes
    node_df = []
    node_df.append(
        pd.DataFrame(
            {
                "label": ss_df["animal"].unique().tolist()
                + ss_df["evt"].unique().tolist()
                + ss_df["evt-sign"].dropna().unique().tolist(),
                "color": "rgb(128,128,128)",
            }
        )
    )
    node_df.append(
        pd.DataFrame(
            {
                "label": ss_df["region"].unique().tolist(),
                "color": map_colors(
                    pd.Series(ss_df["region"].unique()),
                    cc=qualitative.Plotly,
                    return_colors=True,
                ),
            }
        )
    )
    node_df = pd.concat(node_df, ignore_index=True).reset_index()
    # node_df["color"] = convert_colors_to_same_type(node_df["color"].tolist())[0]
    node_ids = node_df.set_index("label")["index"].to_dict()
    node_cols = node_df.set_index("label")["color"].to_dict()
    link_df = []
    # animal - region links
    anm_lk = (
        ss_df.groupby(["animal", "region"])["unit_id"]
        .nunique()
        .reset_index()
        .rename(columns={"unit_id": "value"})
    )
    anm_lk["source"] = anm_lk["animal"].map(node_ids)
    anm_lk["target"] = anm_lk["region"].map(node_ids)
    anm_lk["color"] = anm_lk["region"].map(node_cols)
    link_df.append(anm_lk)
    # reg - event links
    reg_lk = (
        sig_df[sig_df["rej"]]
        .groupby(["region", "evt"])["unit_id"]
        .nunique()
        .reset_index()
        .rename(columns={"unit_id": "value"})
    )
    reg_lk["source"] = reg_lk["region"].map(node_ids)
    reg_lk["target"] = reg_lk["evt"].map(node_ids)
    reg_lk["color"] = reg_lk["region"].map(node_cols)
    link_df.append(reg_lk)
    # event - sign links
    evt_lk = (
        sig_df[sig_df["rej"]]
        .groupby(["region", "evt", "evt-sign"])["unit_id"]
        .nunique()
        .reset_index()
        .rename(columns={"unit_id": "value"})
    )
    evt_lk["source"] = evt_lk["evt"].map(node_ids)
    evt_lk["target"] = evt_lk["evt-sign"].map(node_ids)
    evt_lk["color"] = evt_lk["region"].map(node_cols)
    link_df.append(evt_lk)
    # build plot
    link_df = pd.concat(link_df, ignore_index=True)
    link_df["color"] = link_df["color"].apply(add_color_opacity, alpha=0.6)
    fig = go.Figure(
        data=[
            go.Sankey(
                valueformat=":d",
                valuesuffix=" cells",
                node={
                    "pad": 15,
                    "thickness": 15,
                    "line": {"color": "black", "width": 0.5},
                    "label": node_df["label"],
                    "color": node_df["color"],
                },
                link={
                    "source": link_df["source"],
                    "target": link_df["target"],
                    "value": link_df["value"],
                    "color": link_df["color"],
                },
            )
        ]
    )
    fig.write_html(os.path.join(fig_path, "{}-by_{}.html".format(ss, by)))
    fig.update_layout(autosize=False, height=900, width=1300)
    fig.write_image(os.path.join(fig_path, "{}-by_{}.svg".format(ss, by)))
