import numpy as np
import lightgbm as lgb
from scipy import sparse
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from ..utils import balanced_weights
import warnings
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
    category=UserWarning,
)

GRID = dict(
    num_leaves    = [31, 63],
    max_depth     = [-1],
    learning_rate = [0.05],
)
N_ESTIM = 300
SEED = 42
VAL_FRAC = 0.20
EARLY_STOP = 30

class LightGBMModel:
    def __init__(self):
        pass

    def _stack(self, df, X_cache, tag):
        return sparse.vstack([X_cache[tag][pid] for pid in df.id])

    def inner_select(self, tr_df, X_cache, tag):
        X = self._stack(tr_df, X_cache, tag)
        y = tr_df.y.values
        g = tr_df.gender.values

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        best, best_score = None, -1.0
        for nl in GRID["num_leaves"]:
            for md in GRID["max_depth"]:
                for lr in GRID["learning_rate"]:
                    scores = []
                    for tr_idx, va_idx in skf.split(X, y):
                        wtr = balanced_weights(y[tr_idx], g[tr_idx])
                        m = lgb.LGBMClassifier(
                            n_estimators=N_ESTIM,
                            num_leaves=nl,
                            max_depth=md,
                            learning_rate=lr,
                            objective="binary",
                            metric="binary_logloss",
                            random_state=SEED,
                            n_jobs=1,
                            verbosity=-1,
                        )
                        m.fit(X[tr_idx], y[tr_idx],
                              sample_weight=wtr,
                              eval_set=[(X[va_idx], y[va_idx])],
                              eval_metric="binary_logloss",
                              callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
                        scores.append(balanced_accuracy_score(y[va_idx], m.predict(X[va_idx])))
                    mean = float(np.mean(scores))
                    if mean > best_score:
                        best_score = mean
                        best = ("lightgbm", tag, {"num_leaves": nl, "max_depth": md, "learning_rate": lr})
        return best

    def train_and_eval(self, tr_df, te_df, X_cache, best):
        _, tag, hp = best
        Xtr = self._stack(tr_df, X_cache, tag)
        Xte = self._stack(te_df, X_cache, tag)
        ytr, yte = tr_df.y.values, te_df.y.values
        wtr = balanced_weights(ytr, tr_df.gender.values)

        m = lgb.LGBMClassifier(
            n_estimators=N_ESTIM, **hp,
            objective="binary", metric="binary_logloss",
            random_state=SEED, n_jobs=1, verbosity=-1
        )
        m.fit(Xtr, ytr, sample_weight=wtr)
        y_pred = m.predict(Xte)
        y_score = m.predict_proba(Xte)[:, 1]
        return yte, y_pred, y_score

