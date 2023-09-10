import itertools as itt
import os
import warnings

import numpy as np
import pandas as pd
import xarray as xr
from scipy.io import loadmat
from tqdm.auto import tqdm

DAY_DICT = {
    "Acc1": "Day1",
    "Train1": "Day1",
    "Acc2": "Day2",
    "Train2": "Day2",
    "Acc3": "Day3",
    "Train3": "Day3",
    "Acc4": "Day4",
    "Train4": "Day4",
    "Acc5": "Day5",
    "Test": "Day5",
}
COL_DICT = {"Day4_AccDay4_Test": "Day4"}


def load_mat_data(dpath):
    try:
        cellmap = pd.read_excel(
            os.path.join(dpath, "CellMaps.xlsx"),
            sheet_name=None,
            converters={"Region": lambda r: r.strip("'")},
        )
        cellmap = {k[3:]: v for k, v in cellmap.items()}
    except FileNotFoundError:
        cellmap = None
    matfiles = list(filter(lambda f: f.endswith(".mat"), os.listdir(dpath)))
    anms = set([fn.split("_")[0] for fn in matfiles])
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
            if cellmap is not None:
                day = DAY_DICT[ss]
                cmap = (
                    cellmap[anm]
                    .rename(columns=COL_DICT)[
                        ["ROI Number", "Fluorophore", "Region", day]
                    ]
                    .copy()
                )
                cmap[day] = cmap[day].map(convert_uid)
                cmap = cmap.dropna().set_index(day)
                reg_dict = cmap["Region"].to_dict()
                roi_dict = cmap["ROI Number"].to_dict()
                cur_act = cur_act.assign_coords(
                    region=cur_act.coords["unit_id"]
                    .to_pandas()
                    .map(reg_dict)
                    .to_xarray()
                )
                cur_act = cur_act.assign_coords(
                    roi_id=cur_act.coords["unit_id"]
                    .to_pandas()
                    .map(roi_dict)
                    .to_xarray()
                )
            yield (anm, ss), cur_act, behav_df


def convert_uid(uid):
    try:
        return int(uid)
    except ValueError:
        return np.NAN


def enumerated_product(*args):
    yield from zip(itt.product(*(range(len(x)) for x in args)), itt.product(*args))
