import numpy as np
import pandas as pd


def compute_dff(df, evt_fm_col="evt_fm", act_col="act", return_segments=False):
    evt_sgn = np.sign(df[evt_fm_col].astype(int)) >= 0
    act = df.groupby(evt_sgn)[act_col].mean()
    if return_segments:
        return pd.Series({"pre-evt": act.loc[False], "post-evt": act.loc[True]})
    else:
        return act.loc[True] - act.loc[False]
