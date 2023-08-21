import os
import warnings

import numpy as np
import pandas as pd
import xarray as xr
from scipy.io import loadmat
from tqdm.auto import tqdm


def load_mat_data(dpath):
    anms = set([fn.split("_")[0] for fn in os.listdir(dpath)])
    for anm in tqdm(anms, desc="animal"):
        act = loadmat(
            os.path.join(dpath, "{}_NeuralActivity.mat".format(anm)),
            simplify_cells=True,
        )
        behav = loadmat(
            os.path.join(dpath, "{}_Behavior.mat".format(anm)), simplify_cells=True
        )
        act_dict = dict()
        for ss, a in act.items():
            if ss.startswith("_"):
                continue
            act_dict[ss] = xr.DataArray(
                a,
                dims=["unit_id", "frame"],
                coords={
                    "unit_id": np.arange(a.shape[0]),
                    "frame": np.arange(a.shape[1]),
                    "animal": anm,
                    "session": ss,
                },
                name="act",
            )
        for ss, cur_act in tqdm(act_dict.items(), desc="session", leave=False):
            if ss.startswith("Acc"):
                ss_type = "Acc"
            elif ss.startswith("Train"):
                ss_type = "Train"
            else:
                assert ss == "Test"
                ss_type = "Test"
            try:
                header = behav["{}_Beh".format(ss_type)]
            except:
                header = behav["{}_BehName".format(ss_type)]
            if ss == "Test":
                header0 = np.array(
                    list(
                        map(
                            lambda s: s + "-" if isinstance(s, str) else "",
                            header[:, 1].tolist(),
                        )
                    )
                ).astype(str)
                header1 = np.array(header[:, 0]).astype(str)
                header = np.char.add(header0, header1)
            try:
                behav_df = pd.DataFrame(behav[ss].T, columns=header)
            except ValueError:
                warnings.warn("Cannot load data for {}, {}".format(anm, ss))
                continue
            behav_df["frame"] = np.arange(len(behav_df))
            yield (anm, ss), cur_act, behav_df
