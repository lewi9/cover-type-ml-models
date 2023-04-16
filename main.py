import numpy as np
from src.utils import get_data, train_model
from src.model_heuristic import HeuristicModel
from src.model_RF import RFModel
from src.model_hgb import HistGradientBoostingModel
from sklearn.metrics import classification_report

from sklearn.model_selection import GridSearchCV
if __name__ == "__main__":
    data = get_data()

    check = True
    
    model_heuristic = HeuristicModel()
    train_model(model_heuristic, data, "heuristic", check=check)

    model_RF = RFModel(n_estimators=10, class_weight="balanced",
                       n_jobs=2, verbose=1, random_state=42)
    train_model(model_RF, data, "random_forrest", check=check)


    model_hgb = HistGradientBoostingModel(class_weight="balanced",
                                          verbose=1, random_state=42)
    train_model(model_hgb, data, "hgb", check=check)
