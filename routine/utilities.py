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

PARAM_BEHAV_MAP = {
    "Familiar-BeingAttacked": "attack",
    "Novel-BeingAttacked": "attack",
    "BeingAttacked": "attack",
    "Familiar-SideAttack": "attack",
    "Novel-SideAttack": "attack",
    "SideAttack": "attack",
    "Familiar-AggressiveChase": "aggression",
    "Novel-AggressiveChase": "aggression",
    "AggressiveChase": "aggression",
    "Familiar-Groupsniff_20fps": "sniff",
    "Novel-Groupsniff_20fps": "sniff",
    "Groupsniff_20fps": "sniff",
    "Familiar-SideSniff1_20fps": "sniff",
    "Novel-SideSniff1_20fps": "sniff",
    "SideSniff1_20fps": "sniff",
    "Familiar-dig_20fps": "social",
    "Novel-dig_20fps": "social",
    "dig_20fps": "social",
    "Familiar-nose2nose1_20fpsr": "sniff",
    "Novel-nose2nose1_20fpsr": "sniff",
    "nose2nose1_20fpsr": "sniff",
    "Familiar-nose2rear1_20fps": "sniff",
    "Novel-nose2rear1_20fps": "sniff",
    "nose2rear1_20fps": "sniff",
    "Familiar-nosey1_20fps": "sniff",
    "Novel-nosey1_20fps": "sniff",
    "nosey1_20fps": "sniff",
    "Familiar-follow1_20fps": "social",
    "Novel-follow1_20fps": "social",
    "follow1_20fps": "social",
    "Familiar-groomfront_20fps": "groom",
    "Novel-groomfront_20fps": "groom",
    "groom_front": "groom",
    "groomfront_20fps": "groom",
    "sitcorner": "still",
    "sitcorner_20fps": "still",
    "wallclimb": "move",
    "wallclimb_20fps": "move",
    "walk": "move",
    "walk_20fps": "move",
    "still": "still",
    "still_20fps": "still",
}


def behav_key(k):
    return list(map(list(PARAM_BEHAV_MAP.keys()).index, k))


def classify_behav(row):
    fm = int(row["frame"])
    row = row.drop("frame").sort_index(key=behav_key)
    if row.max() == 1:
        evt = row.idxmax()
        evt_mp = PARAM_BEHAV_MAP[evt]
        if evt.endswith("_20fps"):
            evt = evt[:-6]
        if evt.endswith("_20fpsr"):
            evt = evt[:-7]
        if evt.startswith("Novel-"):
            evt = evt.split("-")[1]
            tgt = "novel"
        elif evt.startswith("Familiar-"):
            evt = evt.split("-")[1]
            tgt = "familiar"
        else:
            tgt = "self"
    else:
        evt = np.nan
        evt_mp = np.nan
        tgt = np.nan
    return pd.Series({"frame": fm, "event": evt, "target": tgt, "event_map": evt_mp})


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
