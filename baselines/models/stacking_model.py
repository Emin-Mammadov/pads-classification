# baselines/models/stacking_model.py
"""
Two-branch stacking baseline:
- Movement branch: LogisticRegression(L1) on BOSS features (tag, default 'all')
- Questionnaire branch: CatBoostClassifier on questionnaire vector
- Meta-learner: LogisticRegression on [p_movement, p_questionnaire]

Inner selection (5-fold):
  - Choose movement C from C_GRID
  - Choose questionnaire depth from CB_DEPTH_GRID
Outer evaluation:
  - Train base models on full outer-train with best HPs
  - Predict probs on outer-test, stack, predict via meta-learner
Returns (y_true, y_pred, y_score) for unified metrics in cv.py.
"""

from __future__ import annotations
import numpy as np
from scipy import sparse

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.linear_model import LogisticRegression

from catboost import CatBoostClassifier

from ..utils import balanced_weights
from ..data import load_questionnaire_vec

C_GRID = (0.1, 1.0, 10.0)
CB_DEPTH_GRID = (4, 6)

SEED = 42


def _load_quest_batch(ids):
    # returns dense 2D float32 array [n, Q]
    return np.vstack([load_questionnaire_vec(pid) for pid in ids]).astype(np.float32, copy=False)


def _stack_boss(tag: str, ids, X_cache_by_tag):
    # X_cache_by_tag[tag][pid] are 1xD CSR rows; vstack keeps CSR
    return sparse.vstack([X_cache_by_tag[tag][pid] for pid in ids])


class StackingModel:
    """
    Implements the model interface expected by baselines.cv.run_outer_cv:
      - inner_select(tr_df, X_cache, tag) -> ("stack", tag, hp_dict)
      - train_and_eval(tr_df, te_df, X_cache, best) -> (y_true, y_pred, y_score)
    """

    name = "stack"
    def __init__(self):
        self._last_selected_hp = None

    # ---------------------- inner selection ----------------------
    def inner_select(self, tr_df, X_cache_by_tag, tag: str):
        """
        Inner 5-fold CV on outer-train to pick:
          - movement C (L1-SAGA LR on BOSS[tag])
          - questionnaire CatBoost depth
        Criterion: mean balanced accuracy across inner folds.
        """
        y = tr_df.y.values
        g = tr_df.gender.values
        ids = tr_df.id.values

        X_mov_full = _stack_boss(tag, ids, X_cache_by_tag)        # sparse CSR
        X_quest_full = _load_quest_batch(ids)                      # dense

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

        # --- choose movement C ---
        best_c, best_c_mean = None, -1.0
        for C in C_GRID:
            scores = []
            for tr_idx, va_idx in skf.split(np.zeros(len(y)), y):
                Xtr = X_mov_full[tr_idx]; ytr = y[tr_idx]
                Xva = X_mov_full[va_idx]; yva = y[va_idx]
                wtr = balanced_weights(ytr, g[tr_idx])

                mov = LogisticRegression(
                    penalty="l1", solver="saga", C=C,
                    class_weight="balanced",
                    max_iter=2000, tol=1e-3, random_state=SEED, n_jobs=-1
                )
                mov.fit(Xtr, ytr, sample_weight=wtr)
                p = mov.predict_proba(Xva)[:, 1]
                scores.append(balanced_accuracy_score(yva, (p > 0.5).astype(int)))
            m = float(np.mean(scores))
            if m > best_c_mean:
                best_c_mean, best_c = m, C

        # --- choose CatBoost depth ---
        best_depth, best_cb_mean = None, -1.0
        for d in CB_DEPTH_GRID:
            scores = []
            for tr_idx, va_idx in skf.split(np.zeros(len(y)), y):
                Xtr = X_quest_full[tr_idx]; ytr = y[tr_idx]
                Xva = X_quest_full[va_idx]; yva = y[va_idx]
                wtr = balanced_weights(ytr, g[tr_idx])

                cb = CatBoostClassifier(
                    iterations=500, depth=d,
                    loss_function="Logloss",
                    eval_metric="BalancedAccuracy",
                    random_state=SEED,
                    early_stopping_rounds=50,
                    thread_count=-1, verbose=False
                )
                cb.fit(Xtr, ytr, sample_weight=wtr, eval_set=(Xva, yva))
                # use hard labels for BA parity with movement branch
                yhat = (cb.predict_proba(Xva)[:, 1] > 0.5).astype(int)
                scores.append(balanced_accuracy_score(yva, yhat))
            m = float(np.mean(scores))
            if m > best_cb_mean:
                best_cb_mean, best_depth = m, d

        hp = {"C_mov": best_c, "depth_quest": best_depth}
        self._last_selected_hp = hp.copy()
        return (self.name, tag, hp)

    # ---------------------- outer train/eval ----------------------
    def train_and_eval(self, tr_df, te_df, X_cache_by_tag, best):
        """
        Train meta-learner using OOF probabilities on outer-train with the
        selected base hyperparameters. Retrain base models on full outer-train,
        predict probabilities on outer-test, and return (y_true, y_pred, y_score).
        """
        _, tag, hp = best
        C_mov = hp["C_mov"]
        d_q = hp["depth_quest"]

        # ----- Prepare outer-train data -----
        y_tr = tr_df.y.values
        g_tr = tr_df.gender.values
        ids_tr = tr_df.id.values

        X_mov_tr = _stack_boss(tag, ids_tr, X_cache_by_tag)   # sparse
        X_quest_tr = _load_quest_batch(ids_tr)                # dense
        w_tr_full = balanced_weights(y_tr, g_tr)

        # Generate OOF probabilities via inner 5-fold (on outer-train)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        oof_mov = np.zeros(len(tr_df), dtype=np.float32)
        oof_quest = np.zeros(len(tr_df), dtype=np.float32)

        for tr_idx, va_idx in skf.split(np.zeros(len(y_tr)), y_tr):
            # Movement
            mov = LogisticRegression(
                penalty="l1", solver="saga", C=C_mov,
                class_weight="balanced",
                max_iter=2000, tol=1e-3, random_state=SEED, n_jobs=-1
            )
            mov.fit(X_mov_tr[tr_idx], y_tr[tr_idx],
                    sample_weight=balanced_weights(y_tr[tr_idx], g_tr[tr_idx]))
            oof_mov[va_idx] = mov.predict_proba(X_mov_tr[va_idx])[:, 1]

            # Questionnaire
            cb = CatBoostClassifier(
                iterations=500, depth=d_q,
                loss_function="Logloss",
                eval_metric="BalancedAccuracy",
                random_state=SEED,
                thread_count=-1, verbose=False
            )
            cb.fit(X_quest_tr[tr_idx], y_tr[tr_idx],
                   sample_weight=balanced_weights(y_tr[tr_idx], g_tr[tr_idx]))
            oof_quest[va_idx] = cb.predict_proba(X_quest_tr[va_idx])[:, 1]

        # Meta-learner trained on OOF probabilities
        P_tr = np.column_stack([oof_mov, oof_quest])
        meta = LogisticRegression(
            solver="liblinear", max_iter=1000,
            class_weight="balanced", random_state=SEED
        )
        meta.fit(P_tr, y_tr)

        # ----- Retrain base models on full outer-train -----
        final_mov = LogisticRegression(
            penalty="l1", solver="saga", C=C_mov,
            class_weight="balanced",
            max_iter=2000, tol=1e-3, random_state=SEED, n_jobs=-1
        )
        final_mov.fit(X_mov_tr, y_tr, sample_weight=w_tr_full)

        final_cb = CatBoostClassifier(
            iterations=500, depth=d_q,
            loss_function="Logloss",
            eval_metric="BalancedAccuracy",
            random_state=SEED,
            thread_count=-1, verbose=False
        )
        final_cb.fit(X_quest_tr, y_tr, sample_weight=w_tr_full)

        # ----- Outer-test -----
        ids_te = te_df.id.values
        y_te = te_df.y.values

        X_mov_te = _stack_boss(tag, ids_te, X_cache_by_tag)
        X_quest_te = _load_quest_batch(ids_te)

        P_te = np.column_stack([
            final_mov.predict_proba(X_mov_te)[:, 1],
            final_cb.predict_proba(X_quest_te)[:, 1]
        ])

        y_score = meta.predict_proba(P_te)[:, 1]
        y_pred  = (y_score > 0.5).astype(int)
        return y_te, y_pred, y_score
