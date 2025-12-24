# baselines/cv.py
import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score
)

def _compute_metrics(y_true, y_pred, y_score=None):
    out = dict(
        accuracy = accuracy_score(y_true, y_pred),
        balanced_accuracy = balanced_accuracy_score(y_true, y_pred),
        f1 = f1_score(y_true, y_pred, zero_division=0),
        precision = precision_score(y_true, y_pred, zero_division=0),
        recall = recall_score(y_true, y_pred, zero_division=0),
        roc_auc = float("nan"),
    )
    if y_score is not None:
        try:
            out["roc_auc"] = roc_auc_score(y_true, y_score)
        except Exception:
            out["roc_auc"] = float("nan")
    return out

def run_outer_cv(df, X_cache_by_tag, model_obj, task: str = "", tag: str = "", model: str = ""):
    fold_metrics = []
    for fold in sorted(df.fold.unique()):
        print(f"[cv] start fold {fold}")
        tr = df[df.fold != fold].reset_index(drop=True)
        te = df[df.fold == fold].reset_index(drop=True)

        best = model_obj.inner_select(tr, X_cache_by_tag, tag)

        y_true, y_pred, y_score = model_obj.train_and_eval(tr, te, X_cache_by_tag, best)
        m = _compute_metrics(y_true, y_pred, y_score)
        fold_metrics.append(m)

        print(f"{task:10s} | {model:12s} | {tag:12s} | fold {fold:2d} "
              f"BA={m['balanced_accuracy']:.4f}  Acc={m['accuracy']:.4f}  "
              f"F1={m['f1']:.4f}  P={m['precision']:.4f}  R={m['recall']:.4f}  "
              f"AUC={m['roc_auc'] if np.isfinite(m['roc_auc']) else float('nan'):.4f}")

    keys = ["balanced_accuracy","accuracy","f1","precision","recall","roc_auc"]
    means = {k: np.nanmean([fm[k] for fm in fold_metrics]) for k in keys}
    print(f"{task:10s} | {model:12s} | {tag:12s} | mean  "
          f"BA={means['balanced_accuracy']:.4f}  Acc={means['accuracy']:.4f}  "
          f"F1={means['f1']:.4f}  P={means['precision']:.4f}  "
          f"R={means['recall']:.4f}  AUC={means['roc_auc']:.4f}")

