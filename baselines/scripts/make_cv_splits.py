import pandas as pd
from sklearn.model_selection import StratifiedKFold
from ..config import cfg

def make_cv_splits():
    """
    Create subject-level 5-fold splits and write to cfg.split_csv.
    Columns: id, fold, label_orig, label_bin
    """
    df = pd.read_csv(cfg.meta_csv, dtype={"id": str})  # expects columns: id,label
    df = df.rename(columns={"label": "label_orig"}).copy()
    df["label_bin"] = (df.label_orig == 1).astype(int)  # PD vs Others

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    df["fold"] = -1
    for f, (_, val_idx) in enumerate(skf.split(df, df.label_bin)):
        df.loc[val_idx, "fold"] = f

    df[["id", "fold", "label_orig", "label_bin"]].to_csv(cfg.split_csv, index=False)
    print(f"cv_splits.csv written: {cfg.split_csv} ({df.shape[0]} rows)")
