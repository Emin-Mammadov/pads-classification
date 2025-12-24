import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

C_VALUES = [0.1, 1, 10]
SEED = 42

class LogisticRegressionModel:
    name = "lr"
    def __init__(self):
        self._last_selected_hp = None

    def _stack(self, df, X_cache, tag):
        # Build X by stacking cached 1xD CSR rows in split order
        return sparse.vstack([X_cache[tag][pid] for pid in df.id])

    def inner_select(self, tr_df, X_cache, tag):
        """
        Inner model selection: **Stratified 5-fold CV** on the outer-train,
        pick C with the best mean Balanced Accuracy. (Parity with your LR script.)
        """
        X = self._stack(tr_df, X_cache, tag)
        y = tr_df.y.values

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        best_C, best_mean = None, -1.0

        for C in C_VALUES:
            scores = []
            for tr_idx, va_idx in skf.split(X, y):
                Xtr, Xva = X[tr_idx], X[va_idx]
                ytr, yva = y[tr_idx], y[va_idx]

                clf = LogisticRegression(
                    penalty="l1",
                    solver="saga",          # supports sparse + L1
                    C=C,
                    class_weight="balanced",
                    max_iter=1500,
                    tol=1e-3,
                    random_state=SEED,
                    n_jobs=-1,
                    warm_start=False,
                )
                clf.fit(Xtr, ytr)
                scores.append(balanced_accuracy_score(yva, clf.predict(Xva)))

            mean_score = float(np.mean(scores))
            if mean_score > best_mean:
                best_mean, best_C = mean_score, C

        self._last_selected_hp = {"C": best_C}
        return ("lr", tag, {"C": best_C})

    def train_and_eval(self, tr_df, te_df, X_cache, best):
        _, tag, hp = best
        Xtr = self._stack(tr_df, X_cache, tag)
        ytr = tr_df.y.values
        Xte = self._stack(te_df, X_cache, tag)
        yte = te_df.y.values

        clf = LogisticRegression(
            penalty="l1", solver="saga", C=hp["C"],
            class_weight="balanced",
            max_iter=1500, tol=1e-3, random_state=SEED, n_jobs=-1,
        )
        clf.fit(Xtr, ytr)
        y_pred = clf.predict(Xte)
        y_score = clf.predict_proba(Xte)[:, 1]
        return yte, y_pred, y_score

