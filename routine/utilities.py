import itertools as itt
import os
import warnings

import numpy as np
import pandas as pd
import xarray as xr
from pandas.api.types import union_categoricals
from scipy.io import loadmat
from tifffile import imread
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
DECONV_PATH = "./intermediate/deconvolution"
BEHAV_THRES_PATH = "./intermediate/behavs_thres/fm_lab_thres.feat"

PARAM_BEHAV_MAP = {
    "Familiar-BeingAttacked": "being_attack",
    "Novel-BeingAttacked": "being_attack",
    "BeingAttacked": "being_attack",
    "Familiar-SideAttack": "attacking",
    "Novel-SideAttack": "attacking",
    "SideAttack": "attacking",
    "Familiar-AggressiveChase": "attacking",
    "Novel-AggressiveChase": "attacking",
    "AggressiveChase": "attacking",
    "Familiar-Groupsniff_20fps": "sniff",
    "Novel-Groupsniff_20fps": "sniff",
    "Groupsniff_20fps": "sniff",
    "Familiar-SideSniff1_20fps": "sniff",
    "Novel-SideSniff1_20fps": "sniff",
    "SideSniff1_20fps": "sniff",
    "Familiar-dig_20fps": "dig",
    "Novel-dig_20fps": "dig",
    "dig_20fps": "dig",
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

RAW_BEHAV_DICT = {
    "Acc1": "CompRawScores_A1",
    "Acc2": "CompRawScores_A2",
    "Acc3": "CompRawScores_A3",
    "Acc4": "CompRawScores_A4",
    "Acc5": "CompRawScores_A5",
    "Train1": "CompRawScores_T1",
    "Train2": "CompRawScores_T2",
    "Train3": "CompRawScores_T3",
    "Train4": "CompRawScores_T4",
    "Test": "CompRawScores_Test",
}


def norm_cells(a, fm_dim="frame"):
    amin, amax = a.min(fm_dim), a.max(fm_dim)
    return (a - amin) / (amax - amin)


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


def parse_behav(evt):
    if pd.isnull(evt):
        return pd.Series({"event": np.nan, "target": np.nan})
    evt_mp = PARAM_BEHAV_MAP[evt]
    if evt.endswith("_20fps"):
        evt = evt[:-6]
    elif evt.endswith("_20fpsr"):
        evt = evt[:-7]
    if evt.startswith("Novel-"):
        evt = evt.split("-")[1]
        tgt = "novel"
    elif evt.startswith("Familiar"):
        evt = evt.split("-")[1]
        tgt = "familiar"
    else:
        tgt = "self"
    return pd.Series({"event": evt_mp, "target": tgt, "event_raw": evt})


def load_mat_data(
    dpath, load_deconv=True, load_act=False, load_roi=False, return_behav="thresholded"
):
    try:
        cellmap = pd.read_excel(
            os.path.join(dpath, "CellMaps.xlsx"), sheet_name=None, header=None
        )
        crossmap = pd.read_feather(os.path.join(dpath, "mapping.feat"))
        cellmap_df = []
        for anm, cmap in cellmap.items():
            cmap.columns = pd.MultiIndex.from_tuples(
                [("meta", "master_id"), ("class", "fluorophore"), ("class", "region")]
            )
            cmap[("meta", "animal")] = anm
            cellmap_df.append(cmap)
        cellmap = pd.concat(cellmap_df)
        cellmap = crossmap.merge(
            cellmap, on=[("meta", "animal"), ("meta", "master_id")], how="left"
        )
    except FileNotFoundError:
        cellmap = None
    anms = [d for d in os.listdir(dpath) if os.path.isdir(os.path.join(dpath, d))]
    for anm in tqdm(anms, desc="animal"):
        anm_path = os.path.join(dpath, anm)
        tif_file = [f for f in os.listdir(anm_path) if f.endswith(".tiff")]
        beh_file = [f for f in os.listdir(anm_path) if f.endswith("_Behavior.mat")]
        raw_file = [f for f in os.listdir(anm_path) if f.endswith("_Raw.mat")]
        assert len(beh_file) == 1, "{} beh file found in {}".format(
            len(beh_file), anm_path
        )
        assert len(raw_file) == 1, "{} raw file found in {}".format(
            len(raw_file), anm_path
        )
        behav = loadmat(os.path.join(dpath, anm, beh_file[0]), simplify_cells=True)
        raw_behav = loadmat(os.path.join(dpath, anm, raw_file[0]), simplify_cells=True)
        for tf in tqdm(tif_file, desc="session", leave=False):
            ss = os.path.splitext(tf)[0].split("-")[1]
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
            if return_behav == "raw":
                behav_df_raw = pd.DataFrame(
                    raw_behav[RAW_BEHAV_DICT[ss]].T, columns=header
                )
                behav_df_raw["frame"] = np.arange(len(behav_df_raw))
                behav_rt = behav_df_raw
            elif return_behav == "thresholded":
                behav_thres = pd.read_feather(BEHAV_THRES_PATH)
                behav_rt = behav_thres[
                    (behav_thres["animal"] == anm) & (behav_thres["session"] == ss)
                ].copy()
            elif return_behav == "binary":
                behav_rt = behav_df
            else:
                raise NotImplementedError(
                    "No Behavior file type: {}".format(return_behav)
                )
            if load_act:
                cur_act = load_tiff(os.path.join(anm_path, tf))
            else:
                cur_act = None
            if load_roi:
                cur_roi = xr.open_dataset(
                    os.path.join(anm_path, "{}.nc".format(os.path.splitext(tf)[0]))
                ).rename({"roi_id": "unit_id"})
            else:
                cur_roi = None
            if load_deconv:
                try:
                    minian_ds = xr.open_dataset(
                        os.path.join(DECONV_PATH, "{}-{}.nc".format(anm, ss))
                    )
                except FileNotFoundError:
                    continue
                curC, curS = minian_ds["C"], minian_ds["S"]
            else:
                curC, curS = None, None
            if cellmap is not None:
                day = DAY_DICT[ss]
                cmap = (
                    cellmap.loc[
                        cellmap["meta", "animal"] == anm,
                        [("session", day), ("meta", "master_id"), ("class", "region")],
                    ]
                    .droplevel(0, axis="columns")
                    .rename(columns={day: "unit_id", "master_id": "roi_id"})
                )
                cmap = cmap.dropna().set_index("unit_id")
                reg_dict = cmap["region"].to_dict()
                roi_dict = cmap["roi_id"].to_dict()
                if load_roi:
                    cur_roi = cur_roi.assign_coords(
                        region=cur_roi.coords["unit_id"]
                        .to_pandas()
                        .map(reg_dict)
                        .to_xarray()
                    )
                    cur_roi = cur_roi.assign_coords(
                        roi_id=cur_roi.coords["unit_id"]
                        .to_pandas()
                        .map(roi_dict)
                        .to_xarray()
                    )
                if load_deconv:
                    curC = curC.assign_coords(
                        region=curC.coords["unit_id"]
                        .to_pandas()
                        .map(reg_dict)
                        .to_xarray()
                    )
                    curC = curC.assign_coords(
                        roi_id=curC.coords["unit_id"]
                        .to_pandas()
                        .map(roi_dict)
                        .to_xarray()
                    )
                    curS = curS.assign_coords(
                        region=curS.coords["unit_id"]
                        .to_pandas()
                        .map(reg_dict)
                        .to_xarray()
                    )
                    curS = curS.assign_coords(
                        roi_id=curS.coords["unit_id"]
                        .to_pandas()
                        .map(roi_dict)
                        .to_xarray()
                    )
            yield (anm, ss), cur_act, curC, curS, cur_roi, behav_rt


def convert_uid(uid):
    try:
        return int(uid)
    except ValueError:
        return np.NAN


def enumerated_product(*args):
    yield from zip(itt.product(*(range(len(x)) for x in args)), itt.product(*args))


def agg_across(df, cols, val, multiple_val=False, return_val=True):
    if multiple_val:
        vals = val
    else:
        vals = [val]
    agg_cols = list(set(df.columns) - set(cols) - set(vals))
    if return_val:
        return df.groupby(agg_cols, observed=True)[val]
    else:
        return df.groupby(agg_cols, observed=True)


def concat_cat(df_ls, cat_cols):
    dtypes = {c: union_categoricals([d[c] for d in df_ls]).dtype for c in cat_cols}
    df_ls = [d.astype(dtypes) for d in df_ls]
    return pd.concat(df_ls, ignore_index=True)


def load_tiff(fpath):
    img = imread(fpath)
    return xr.DataArray(
        img,
        dims=["frame", "height", "width"],
        coords={
            "frame": np.arange(img.shape[0]),
            "height": np.arange(img.shape[1]),
            "width": np.arange(img.shape[2]),
        },
    )
