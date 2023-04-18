import pickle
import numpy as np
import os.path
from src.utils import get_data, train_model, train_model_nn, transform_parameters, save_data, visualize
from src.model_heuristic import HeuristicModel
from src.model_RF import RFModel
from src.model_hgb import HistGradientBoostingModel
from src.model_nn import NNModel


def tune_nn(data, check, pop=6, gen=6, epoch=6, metric="accuracy"):
    """
    Function load or search hyperparameters for neural-network model.
    NSGA2 is used - multicriteria Genetic Algorithm.

    :param data: <pd.DataFrame> with data
    :param check: <bool> check if file with hyperparameters exist
    :param pop: <int> population per generation
    :param gen: <int> number of generations
    :param epoch: <int> number of epoch used to train model
    :param metric: 'loss'|'accuracy'|'f1score' metric to select hyperparameters
    :return: <list> set of hyperparameters coded as ints
    """
    if check:
        if (not os.path.isfile(f"data/hyperparameters_set-final.pkl") or
                not os.path.isfile(f"data/result_of_hyperparameters-final.pkl")):

            result = NNModel.tuning_nn(data, pop=10, gen=20, epoch=5)
            save_data(result.X, "hyperparameters_set-final")
            save_data(result.F, "result_of_hyperparameters-final")

    with (open("data/hyperparameters_set-final.pkl", "rb")) as openfile:
        parameters = pickle.load(openfile)
    with (open("data/result_of_hyperparameters-final.pkl", "rb")) as openfile:
        result = pickle.load(openfile)

    # pareto front is set of non-dominated solutions. Non-dominated solution can't improve (minimize or maximize)
    # one of values [f1, f2, f3] and not to decrease another.

    # example (maximize): in set [[1, 1, 2], [1, 1, 1], [0, 0, 3]], solution [1, 1, 2] and [0, 0, 3] are non-dominated
    # and solution [1, 1, 1] is dominated by first, because we know that we can improve third parameter '1' and not
    # decrease other. So pareto front is [[1, 1, 2], [0, 0, 3]]

    # I try to minimize 3 metrics - loss, 1 - accuracy and 1 - f1_score

    # result is a pareto front, so we can more explore these models to find best or just select one by metric
    # we can also select model that is average in all metrics (not implemented)

    index = np.argmin(result, axis=0)
    if metric == "loss":
        return parameters[index[0]]
    elif metric == "accuracy":
        return parameters[index[1]]
    elif metric == "f1score":
        return parameters[index[2]]
    else:
        raise ValueError("Provide one of valid metrics: ['loss', 'accuracy', 'f1score']")


if __name__ == "__main__":
    # read data
    data = get_data()

    # Set false if you want to create models if they exist
    check = True

    # Create models if they don't exist
    model_heuristic = HeuristicModel()
    train_model(model_heuristic, data, "heuristic", check=check)

    model_RF = RFModel(n_estimators=10, class_weight="balanced", n_jobs=2, verbose=1, random_state=42)
    train_model(model_RF, data, "random_forrest", check=check)

    model_hgb = HistGradientBoostingModel(class_weight="balanced", verbose=1, random_state=42)
    train_model(model_hgb, data, "hgb", check=check)

    parameters = tune_nn(data, check=check, pop=10, gen=10, metric="accuracy")
    model_nn = NNModel(*transform_parameters(parameters[1:]))
    train_model_nn(model_nn, data, "nn", check=check, epoch=150, batch_size=2**parameters[0])

    # Save visualization to ./figures/
    visualize(models=["heuristic", "random_forrest", "hgb", "nn"], data=data)
