import numpy as np
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

# Parity grids
N_ESTIMATORS = [100,300, 500]
MAX_DEPTH    = [5, 10, None]
SEED = 42

class RandomForestModel:
    name = "rf"
    def __init__(self):
        self._last_selected_hp = None

    def _stack(self, df, X_cache, tag):
        # RF in sklearn accepts scipy sparse; keep CSR to avoid needless densification.
        return sparse.vstack([X_cache[tag][pid] for pid in df.id])

    def inner_select(self, tr_df, X_cache, tag):
        """
        Stratified 5-fold inner CV on outer-train to choose (n_estimators, max_depth).
        """
        X = self._stack(tr_df, X_cache, tag)
        y = tr_df.y.values

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        best_params, best_mean = None, -1.0

        for n_est in N_ESTIMATORS:
            for md in MAX_DEPTH:
                scores = []
                for tr_idx, va_idx in skf.split(X, y):
                    Xtr, Xva = X[tr_idx], X[va_idx]
                    ytr, yva = y[tr_idx], y[va_idx]

                    rf = RandomForestClassifier(
                        n_estimators=n_est,
                        max_depth=md,
                        class_weight="balanced",
                        random_state=SEED,
                        n_jobs=1,
                    )
                    rf.fit(Xtr, ytr)
                    scores.append(balanced_accuracy_score(yva, rf.predict(Xva)))

                mean_score = float(np.mean(scores))
                if mean_score > best_mean:
                    best_mean = mean_score
                    best_params = {"n_estimators": n_est, "max_depth": md}
        self._last_selected_hp = best_params.copy()
        return ("rf", tag, best_params)

    def train_and_eval(self, tr_df, te_df, X_cache, best):
        _, tag, hp = best
        Xtr = self._stack(tr_df, X_cache, tag)
        ytr = tr_df.y.values
        Xte = self._stack(te_df, X_cache, tag)
        yte = te_df.y.values

        rf = RandomForestClassifier(
            n_estimators=hp["n_estimators"],
            max_depth=hp["max_depth"],
            class_weight="balanced",
            random_state=SEED, n_jobs=1,
        )
        rf.fit(Xtr, ytr)
        y_pred = rf.predict(Xte)
        y_score = rf.predict_proba(Xte)[:, 1]
        return yte, y_pred, y_score

