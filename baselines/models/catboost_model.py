# baselines/models/catboost_model.py
from catboost import CatBoostClassifier
from scipy import sparse
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from ..utils import balanced_weights
import numpy as np

GRID_DEPTH=[4,6]; GRID_LR=[0.05]; GRID_L2=[3]
N_TREES=300; SEED=42; VAL_FRACTION=0.20
EARLY_STOP = 30

class CatBoostModel:
    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu

    def _new_clf(self, **hp):
        extra = {"task_type": "GPU", "devices": "0"} if self.use_gpu else {}
        return CatBoostClassifier(
            iterations=N_TREES,
            loss_function="Logloss",
            eval_metric="BalancedAccuracy",
            random_state=SEED,
            thread_count=1,
            verbose=False,
            **hp, **extra
        )

    def _stack(self, df, X_cache, tag):
        # Parity: load the cached matrix for exactly this tag
        return sparse.vstack([X_cache[tag][pid] for pid in df.id])

    def inner_select(self, tr_df, X_cache, tag):
        X_full = self._stack(tr_df, X_cache, tag)
        y_full = tr_df.y.values
        g_full = tr_df.gender.values

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        best_m, best_hp = -1.0, None
        for d in GRID_DEPTH:
            for lr in GRID_LR:
                for l2 in GRID_L2:
                    scores = []
                    for tr_idx, va_idx in skf.split(X_full, y_full):
                        w_tr = balanced_weights(y_full[tr_idx], g_full[tr_idx])
                        m = self._new_clf(depth=d, learning_rate=lr, l2_leaf_reg=l2)
                        m.fit(X_full[tr_idx], y_full[tr_idx],
                              sample_weight=w_tr,
                              eval_set=(X_full[va_idx], y_full[va_idx]),
                              early_stopping_rounds=EARLY_STOP)
                        # score via predictions for strict parity with others
                        y_hat = m.predict(X_full[va_idx])
                        scores.append(balanced_accuracy_score(y_full[va_idx], y_hat))
                    mean = float(np.mean(scores))
                    if mean > best_m:
                        best_m = mean
                        best_hp = {"depth": d, "learning_rate": lr, "l2_leaf_reg": l2}
        return ("catboost", tag, best_hp)

    def train_and_eval(self, tr_df, te_df, X_cache, best):
        _, tag, hp = best
        Xtr = self._stack(tr_df, X_cache, tag)
        Xte = self._stack(te_df, X_cache, tag)
        ytr, yte = tr_df.y.values, te_df.y.values
        w = balanced_weights(ytr, tr_df.gender.values)
        m = self._new_clf(**hp)
        m.fit(Xtr, ytr, sample_weight=w)
        y_pred = m.predict(Xte)
        y_score = m.predict_proba(Xte)[:, 1]
        return yte, y_pred, y_score

