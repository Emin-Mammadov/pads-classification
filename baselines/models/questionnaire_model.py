# baselines/models/questionnaire_model.py
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import balanced_accuracy_score
from catboost import CatBoostClassifier
from ..data import load_questionnaire_vec

def _stack(df):
    return np.vstack([load_questionnaire_vec(pid) for pid in df.id]), df.y.values

# Base “interface” for parity with other models
class _QuestionnaireBase:
    # No inner tuning — return fixed HPs; tag is ignored.
    def inner_select(self, tr_df, X_cache, tag):
        return (self.name, "questionnaire", self.hp)

    def train_and_eval(self, tr_df, te_df, X_cache, best):
        Xtr, ytr = _stack(tr_df)
        Xte, yte = _stack(te_df)
        clf = self._build(ytr)   # some need class balance based on ytr
        clf.fit(Xtr, ytr)

        # hard predictions
        y_pred = clf.predict(Xte)

        # scoring for ROC AUC (proba if available, else decision_function, else None)
        if hasattr(clf, "predict_proba"):
            y_score = clf.predict_proba(Xte)[:, 1]
        elif hasattr(clf, "decision_function"):
            y_score = clf.decision_function(Xte)
        else:
            y_score = None

        return yte, y_pred, y_score

class QuestionnaireCatBoost(_QuestionnaireBase):
    name = "quest_catboost"
    uses_boss_features = False
    hp = dict(iterations=300, depth=6, learning_rate=0.1)
    def _build(self, ytr):
        n0 = int((ytr == 0).sum()); n1 = max(1, int((ytr == 1).sum()))
        pos_w = n0 / n1
        return CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="BalancedAccuracy",
            class_weights=[1.0, float(pos_w)],
            thread_count=-1, verbose=False, **self.hp
        )

class QuestionnaireLogReg(_QuestionnaireBase):
    uses_boss_features = False
    name = "quest_lr"
    hp = {"C": 1.0}
    def _build(self, ytr):
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                penalty="l2", solver="liblinear",
                C=self.hp["C"], class_weight="balanced",
                random_state=42
            )
        )

class QuestionnaireSVM(_QuestionnaireBase):
    uses_boss_features = False
    name = "quest_svm"
    hp = {"C": 1.0}
    def _build(self, ytr):
        return make_pipeline(
            StandardScaler(),
            LinearSVC(
                penalty="l2", C=self.hp["C"],
                class_weight="balanced",
                max_iter=5000, random_state=42
            )
        )
