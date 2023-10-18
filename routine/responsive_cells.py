import numpy as np


def compute_dff(df, evt_fm_col="evt_fm", act_col="act"):
    evt_sgn = np.sign(df[evt_fm_col].astype(int)) >= 0
    act = df.groupby(evt_sgn)[act_col].mean()
    return act.loc[True] - act.loc[False]
