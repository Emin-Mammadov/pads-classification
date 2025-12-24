import time
import numpy as np
from scipy import sparse
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

HIDDEN_SIZES = [(50,), (100,), (50, 50)]
LEARNING_RATE_INITS = [0.001, 0.01]
MAX_ITERS = [200, 500]
SEED = 42

class MLPModel:
    name = "mlp"
    def __init__(self):
        self._last_selected_hp = None

    def _stack_dense(self, df, X_cache, tag):
        # MLP needs dense arrays; keep memory in check with float32
        X = sparse.vstack([X_cache[tag][pid] for pid in df.id])
        return X.toarray().astype(np.float32, copy=False)

    def inner_select(self, tr_df, X_cache, tag):
        """
        Inner model selection: Stratified 5-fold CV on the outer-train.
        Choose (hidden_layer_sizes, learning_rate_init, max_iter) with best mean BA.
        """
        X = self._stack_dense(tr_df, X_cache, tag)
        y = tr_df.y.values

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        best_params, best_mean = None, -1.0

        for hs in HIDDEN_SIZES:
            for lr in LEARNING_RATE_INITS:
                for mi in MAX_ITERS:
                    scores = []
                    for tr_idx, va_idx in skf.split(X, y):
                        Xtr, Xva = X[tr_idx], X[va_idx]
                        ytr, yva = y[tr_idx], y[va_idx]

                        mlp = MLPClassifier(
                            hidden_layer_sizes=hs,
                            learning_rate_init=lr,
                            max_iter=mi,
                            alpha=0.0001,
                            solver="adam",
                            random_state=SEED,
                        )
                        mlp.fit(Xtr, ytr)
                        scores.append(balanced_accuracy_score(yva, mlp.predict(Xva)))

                    mean_score = float(np.mean(scores))
                    if mean_score > best_mean:
                        best_mean = mean_score
                        best_params = {"hidden_layer_sizes": hs,
                                       "learning_rate_init": lr,
                                       "max_iter": mi}
        self._last_selected_hp = best_params.copy()
        return ("mlp", tag, best_params)

    def train_and_eval(self, tr_df, te_df, X_cache, best):
        _, tag, hp = best
        Xtr = self._stack_dense(tr_df, X_cache, tag)
        ytr = tr_df.y.values
        Xte = self._stack_dense(te_df, X_cache, tag)
        yte = te_df.y.values

        mlp = MLPClassifier(
            hidden_layer_sizes=hp["hidden_layer_sizes"],
            learning_rate_init=hp["learning_rate_init"],
            max_iter=hp["max_iter"],
            alpha=0.0001, solver="adam", random_state=SEED,
        )
        mlp.fit(Xtr, ytr)
        y_pred = mlp.predict(Xte)
        y_score = mlp.predict_proba(Xte)[:, 1]
        return yte, y_pred, y_score

