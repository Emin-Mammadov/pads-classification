# baselines/models/__init__.py
from .catboost_model import CatBoostModel
from .lightgbm_model import LightGBMModel
from .logreg_model import LogisticRegressionModel
from .mlp_model import MLPModel
from .rf_model import RandomForestModel
from .stacking_model import StackingModel
from .svm_model import SVMModel
from .questionnaire_model import QuestionnaireCatBoost, QuestionnaireLogReg, QuestionnaireSVM

REGISTRY = {
    "catboost": CatBoostModel,
    "lightgbm": LightGBMModel,
    "lr":       LogisticRegressionModel,
    "mlp":      MLPModel,
    "rf":       RandomForestModel,
    "svm":      SVMModel,
    "stack": StackingModel,
    # questionnaire (single file, three entries)
    "quest_catboost": QuestionnaireCatBoost,
    "quest_lr":       QuestionnaireLogReg,
    "quest_svm":      QuestionnaireSVM,
}
