import numpy as np
from scipy import sparse
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

SEED = 42
C_GRID = [0.01, 0.1, 1, 10]

def _pos_class_weight(y):
    """class_weight for LinearSVC"""
    n0 = int((y == 0).sum())
    n1 = int((y == 1).sum())
    return {0: 1.0, 1: (n0 / n1) if n1 > 0 else 1.0}

class SVMModel:
    name = "svm"
    def __init__(self):
        self._last_selected_hp = None

    def _stack(self, df, X_cache, tag):
        # Stack cached 1xD CSR rows in split order (keeps it sparse).
        return sparse.vstack([X_cache[tag][pid] for pid in df.id])

    def inner_select(self, tr_df, X_cache, tag):
        """
        Inner model selection: Stratified 5-fold CV on outer-train,
        pick C with best mean Balanced Accuracy.
        """
        X = self._stack(tr_df, X_cache, tag)
        y = tr_df.y.values

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        best_C, best_mean = None, -1.0

        for C in C_GRID:
            scores = []
            for tr_idx, va_idx in skf.split(X, y):
                Xtr, Xva = X[tr_idx], X[va_idx]
                ytr, yva = y[tr_idx], y[va_idx]

                cw = _pos_class_weight(ytr)

                # StandardScaler(with_mean=False) is sparse-safe
                clf = make_pipeline(
                    StandardScaler(with_mean=False),
                    LinearSVC(
                        C=C, class_weight=cw, dual=False,
                        max_iter=5000, random_state=SEED
                    )
                )
                clf.fit(Xtr, ytr)
                scores.append(balanced_accuracy_score(yva, clf.predict(Xva)))

            mean_score = float(np.mean(scores))
            if mean_score > best_mean:
                best_mean, best_C = mean_score, C
        self._last_selected_hp = {"C": best_C}
        return ("svm", tag, {"C": best_C})

    def train_and_eval(self, tr_df, te_df, X_cache, best):
        _, tag, hp = best
        Xtr = self._stack(tr_df, X_cache, tag)
        ytr = tr_df.y.values
        Xte = self._stack(te_df, X_cache, tag)
        yte = te_df.y.values

        cw = _pos_class_weight(ytr)
        clf = make_pipeline(
            StandardScaler(with_mean=False),
            LinearSVC(
                C=hp["C"], class_weight=cw, dual=False,
                max_iter=5000, random_state=SEED
            )
        )
        clf.fit(Xtr, ytr)
        y_pred = clf.predict(Xte)
        # pipeline exposes decision_function on the final step
        y_score = clf.decision_function(Xte)
        return yte, y_pred, y_score

