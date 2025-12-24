# data.py
import json
import joblib, numpy as np, pandas as pd
from scipy.sparse import csr_matrix
from .config import cfg

def read_split():
    try: return pd.read_csv(cfg.split_csv, dtype={"id": str})
    except pd.errors.ParserError: return pd.read_csv(cfg.split_csv, sep=";", dtype={"id": str})

def read_meta(cols=("id","label","gender")):
    return pd.read_csv(cfg.meta_csv, dtype={"id": str})[list(cols)]

def load_vec(pid: str, tag: str) -> csr_matrix:
    arr = joblib.load(cfg.feat_dir / f"{pid}_{tag}.joblib")
    return csr_matrix(arr.reshape(1,-1))

def map_labels(df, task):
    if task == "pd_vs_hc":
        df = df[df.label.isin([0,1])].copy(); df["y"] = df.label.values
    elif task == "pd_vs_dd":
        df = df[df.label.isin([1,2])].copy(); df["y"] = df.label.map({1:0,2:1}).values
    else: raise ValueError(task)
    return df

def load_questionnaire_vec(pid: str) -> np.ndarray:
    fp = cfg.root / "questionnaire" / f"questionnaire_response_{pid}.json"
    with open(fp, "r") as f:
        js = json.load(f)
    items = sorted(js["item"], key=lambda d: int(d["link_id"]))
    # 1.0 if answered True-ish, else 0.0
    return np.array([1.0 if it.get("answer", False) else 0.0 for it in items],
                    dtype=np.float32)