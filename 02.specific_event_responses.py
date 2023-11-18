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

from routine.plotting import add_color_opacity, facet_plotly, map_colors
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
ord_dict = {
    "Active": 0,
    "Social: General": 1,
    "Social: Familiar": 2,
    "Social: Novel": 3,
    "Aggressor": 4,
    "Aggressor: Familiar": 5,
    "Aggressor: Novel": 6,
    "Target of Aggression": 7,
    "NonSocial": 8,
    "NonActive": 9,
}


def arrange_ypos(sizes, pad=2e-2):
    prop = np.array(sizes / sizes.sum() * (1 - pad * (len(sizes) + 1)))
    pos = np.zeros_like(prop, dtype=float)
    pos[0] = pad + prop[0] / 2
    for i in range(1, len(pos)):
        pos[i] = pos[i - 1] + prop[i - 1] / 2 + prop[i] / 2 + pad
    return pos


def sort_labs(labs):
    return labs.map(val_index)


def val_index(lab):
    if lab.endswith("-activated"):
        add_val = 0
    elif lab.endswith("-suppressed"):
        add_val = 1
    else:
        add_val = 0
    try:
        return (
            ord_dict[lab.replace("-activated", "").replace("-suppressed", "")] * 100
            + add_val
        )
    except KeyError:
        return lab


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
    anm_nd = pd.DataFrame(
        {
            "label": ss_df["animal"].unique().tolist(),
            "color": "rgb(128,128,128)",
            "node_type": "animal",
        }
    )
    evt_nd = pd.DataFrame(
        {
            "label": ss_df["evt"].unique().tolist(),
            "color": "rgb(128,128,128)",
            "node_type": "evt",
        }
    )
    evt_sign_nd = pd.DataFrame(
        {
            "label": ss_df["evt-sign"].dropna().unique().tolist(),
            "color": "rgb(128,128,128)",
            "node_type": "evt_sign",
        }
    )
    reg_nd = pd.DataFrame(
        {
            "label": ss_df["region"].unique().tolist(),
            "color": map_colors(
                pd.Series(ss_df["region"].unique()),
                cc=qualitative.Plotly,
                return_colors=True,
            ),
            "node_type": "reg",
        }
    )
    node_dfs = {
        "full": pd.concat([anm_nd, evt_nd, evt_sign_nd, reg_nd], ignore_index=True)
        .sort_values(
            ["node_type", "label"],
            key=sort_labs,
        )
        .reset_index(drop=True)
        .reset_index(),
        "simple": pd.concat([evt_nd, reg_nd], ignore_index=True)
        .sort_values(
            ["node_type", "label"],
            key=sort_labs,
        )
        .reset_index(drop=True)
        .reset_index(),
    }
    for plt_type, node_df in node_dfs.items():
        node_ids = node_df.set_index("label")["index"].to_dict()
        node_cols = node_df.set_index("label")["color"].to_dict()
        link_df = []
        # animal - region links
        if plt_type == "full":
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
        if plt_type == "full":
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
        sizes = (
            pd.concat(
                [
                    link_df.groupby("source")["value"].sum(),
                    link_df.groupby("target")["value"].sum(),
                ],
                axis="columns",
            )
            .max(axis="columns")
            .rename("size")
            .astype(int)
            .reset_index()
        )
        node_df = node_df.merge(sizes, on="index", how="left")
        node_df["y"] = node_df.groupby("node_type")["size"].transform(arrange_ypos)
        node_df["x"] = node_df["node_type"].map(
            {"animal": 1e-6, "reg": 0.33, "evt": 0.66, "evt_sign": 1 - 1e-6}
        )
        fig = go.Figure(
            data=[
                go.Sankey(
                    arrangement="perpendicular",
                    valueformat=":d",
                    valuesuffix=" cells",
                    node={
                        "pad": 15,
                        "thickness": 15,
                        "line": {"color": "black", "width": 0.5},
                        "label": node_df["label"],
                        "color": node_df["color"],
                        "x": node_df["x"],
                        "y": node_df["y"],
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
        fpath = os.path.join(fig_path, plt_type)
        os.makedirs(fpath, exist_ok=True)
        fig.write_html(os.path.join(fpath, "{}-by_{}.html".format(ss, by)))
        fig.update_layout(autosize=False, height=900, width=1300)
        fig.write_image(os.path.join(fpath, "{}-by_{}.svg".format(ss, by)))

# %% build sunburst chart
fig_path = os.path.join(FIG_PATH, "count")
os.makedirs(fig_path, exist_ok=True)
resp_tt = pd.read_feather(os.path.join(OUT_PATH, "resp_tt.feat"))
resp_tt["rej"] = resp_tt["rej"].where(resp_tt["rej_any"], False)
rej_count = (
    resp_tt.groupby(["animal", "session", "by", "region", "unit_id"], observed=True)[
        "rej"
    ]
    .sum()
    .rename("rej_count")
    .reset_index()
)
rej_df = (
    rej_count.groupby(
        ["animal", "session", "by", "region", "rej_count"], observed=True
    )["unit_id"]
    .count()
    .rename("ncell")
    .reset_index()
)
anm_df = (
    rej_count.groupby(["session", "by", "region", "rej_count"], observed=True)[
        "unit_id"
    ]
    .count()
    .rename("ncell")
    .reset_index()
)
anm_df["animal"] = "ALL"
rej_df = pd.concat([anm_df, rej_df], ignore_index=True).sort_values(
    ["by", "animal", "session", "region", "rej_count"]
)
for by, by_df in rej_df.groupby("by", observed=True):
    fig, layout = facet_plotly(
        by_df, facet_row="session", facet_col="animal", specs={"type": "sunburst"}
    )
    for (ss, anm), ssdf in by_df.groupby(["session", "animal"]):
        ly = layout.loc[(ss, anm)]
        row, col = ly["row"] + 1, ly["col"] + 1
        lv0 = (
            ssdf.groupby(["animal", "region"], observed=True)["ncell"]
            .sum()
            .reset_index()
            .rename(columns={"animal": "parent", "region": "label", "ncell": "value"})
        )
        lv0["name"] = lv0["label"]
        lv1 = ssdf[["region", "rej_count", "ncell"]].rename(
            columns={"region": "parent", "rej_count": "label", "ncell": "value"}
        )
        lv1["name"] = lv1["parent"].astype(str) + "-" + lv1["label"].astype(str)
        plt_df = pd.concat(
            [
                lv0,
                lv1,
                pd.DataFrame(
                    [
                        {
                            "parent": "",
                            "label": anm,
                            "value": ssdf["ncell"].sum(),
                            "name": anm,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
        fig.add_trace(
            go.Sunburst(
                ids=plt_df["name"],
                labels=plt_df["label"],
                parents=plt_df["parent"],
                values=plt_df["value"],
                branchvalues="total",
                hovertemplate="%{label}: %{value} cells",
                name="-".join([anm, ss]),
            ),
            row=row,
            col=col,
        )
    fig.update_layout(height=6000)
    fig.write_html(os.path.join(fig_path, "by_{}.html".format(by)))
    fig.update_layout(height=6000, width=2000)
    fig.write_image(os.path.join(fig_path, "by_{}.svg".format(by)))
