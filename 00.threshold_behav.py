# %% imports and definition
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from scipy.stats import zscore

from routine.utilities import load_mat_data

IN_DPATH = "./data"
OUT_PATH = "./intermediate/behavs_thres"
FIG_PATH = "./figs/behav_thres"
PARAM_ZTHRES = 2
PARAM_SIGTHRES = 2
os.makedirs(FIG_PATH, exist_ok=True)
os.makedirs(OUT_PATH, exist_ok=True)


def get_max_behav(fm_df, behav_col="behavior", metric="zscore"):
    fm_df = (
        fm_df[[behav_col, metric]]
        .dropna()
        .sort_values(metric, ascending=False)
        .reset_index(drop=True)
    )
    if len(fm_df) > 1:
        return pd.Series(
            {
                behav_col: fm_df.loc[0, behav_col],
                metric: fm_df.loc[0, metric],
                "sig": fm_df.loc[0, metric] - fm_df.loc[1, metric],
            }
        )
    else:
        return pd.Series(
            {
                behav_col: fm_df.loc[0, behav_col],
                metric: fm_df.loc[0, metric],
                "sig": np.nan,
            }
        )


# %% load and aggregate events
behavs = []
for (anm, ss), act, curC, curS, behav_df in load_mat_data(IN_DPATH, return_behav="raw"):
    behs = list(set(behav_df.columns) - set(["frame"]))
    for b in behs:
        bdf = behav_df[b].rename("raw").to_frame()
        bdf["frame"] = behav_df["frame"]
        bdf["behavior"] = b
        bdf["animal"] = anm
        bdf["session"] = ss
        behavs.append(bdf)
behavs = pd.concat(behavs, ignore_index=True)
behavs["zscore"] = behavs.groupby("behavior")["raw"].transform(zscore)
behavs.to_feather(os.path.join(OUT_PATH, "scores.feat"))

# %% plot behav score distribution
behavs = pd.read_feather(os.path.join(OUT_PATH, "scores.feat"))
for xvar in ["raw", "zscore"]:
    g = sns.displot(
        data=behavs,
        x=xvar,
        row="behavior",
        col="animal",
        rug=False,
        kind="hist",
        stat="probability",
        log=True,
        facet_kws={
            "sharex": "row",
            "sharey": "row",
            "legend_out": True,
            "margin_titles": True,
        },
        height=3,
        bins=40,
        multiple="dodge",
    )
    fig = g.fig
    fig.savefig(os.path.join(FIG_PATH, "{}.svg".format(xvar)))
    plt.close(fig)

# %% generate frame label
behavs = pd.read_feather(os.path.join(OUT_PATH, "scores.feat"))
behav_lab = (
    behavs.groupby(["animal", "session", "frame"]).apply(get_max_behav).reset_index()
)
behav_lab.to_feather(os.path.join(OUT_PATH, "fm_lab.feat"))

# %% plot scatter
behav_lab = pd.read_feather(os.path.join(OUT_PATH, "fm_lab.feat"))
lab_sub = behav_lab[behav_lab["zscore"] > 0]
fig = px.scatter(
    lab_sub,
    x="zscore",
    y="sig",
    color="behavior",
    facet_row="session",
    facet_col="animal",
)
fig.update_traces(marker={"size": 2})
fig.update_layout(width=1900, height=1900 * 2)
fig.write_html(os.path.join(FIG_PATH, "zscore:sig.html"))

# %% threshold behavior
behav_lab = pd.read_feather(os.path.join(OUT_PATH, "fm_lab.feat"))
behav_lab["behavior"] = behav_lab["behavior"].where(
    (behav_lab["zscore"] > PARAM_ZTHRES) & (behav_lab["sig"] > PARAM_SIGTHRES), np.nan
)
behav_lab.to_feather(os.path.join(OUT_PATH, "fm_lab_thres.feat"))


# %% plot behavior histogram
def make_bar(color, **kwargs):
    ax = sns.barplot(**kwargs)
    ax.set_yscale("log")
    return ax


behav_lab = pd.read_feather(os.path.join(OUT_PATH, "fm_lab_thres.feat"))
behav_ct = (
    behav_lab.groupby(["animal", "session", "behavior"])
    .count()["frame"]
    .rename("count")
    .reset_index()
)
behav_ct = behav_ct[behav_ct["behavior"] != None].copy()
g = sns.FacetGrid(
    data=behav_ct,
    row="session",
    col="animal",
    hue="behavior",
    sharex="row",
    sharey="row",
    legend_out=True,
    margin_titles=True,
    aspect=0.6,
    height=6,
)
g.map_dataframe(make_bar, x="behavior", y="count")
g.set_xticklabels(rotation=90)
fig = g.fig
fig.tight_layout()
os.makedirs(FIG_PATH, exist_ok=True)
fig.savefig(os.path.join(FIG_PATH, "counts.svg"), bbox_inches="tight")
fig.savefig(os.path.join(FIG_PATH, "counts.png"), bbox_inches="tight", dpi=300)
