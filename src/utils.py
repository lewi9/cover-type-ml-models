import urllib.request
import os.path
import pickle
import warnings
import bz2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay


def get_data():
    """
    Download data if doesn't exist, load it.

    :return: <pd.DataFrame> covtype.data
    """
    path = "data/covtype.data.gz"
    if not os.path.isfile(path):
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
        urllib.request.urlretrieve(url, path)
    return pd.read_csv(path, header=None)


def data_split(data):
    """
    Split data into training and testing set.

    :param data: <pd.DataFrame> data to be splitted
    :return: <tuple> X_train, X_test, y_train, y_test
    """
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    return train_test_split(X, y, test_size=0.25, random_state=42,
                            stratify=y, shuffle=True)


def save_model(model, path):
    """
    Function save model to pkl file.

    :param model: HeuristicModel|HistGradientBoostingModel|NNModel|RFModel object of one of model class
    :param path: path to save model
    :return: None
    """
    ofile = bz2.BZ2File(path, 'wb')
    pickle.dump(model, ofile)
    ofile.close()


def save_data(data, name):
    """
    Function dump data into pkl file

    :param data: many types, should be served by pickle
    :param name: <string> name of file
    :return: None
    """
    pickle.dump(data, open(f"data/{name}.pkl", "wb"))


def train_model(model, data, name, check):
    """
    Function train model, fit it and save to file (except neural-network).

    :param model: HeuristicModel|HistGradientBoostingModel|RFModel object of one of model class
    :param data: <pd.DataFrame> covtype.data
    :param name: <string> name of file, model should be saved
    :param check: <bool> check if file with model exists. If exists, return
    :return: None
    """
    if check:
        if os.path.isfile(f"models/{name}.pkl.bz2"):
            return
    np.random.seed(42)
    X_train, X_test, y_train, y_test = data_split(data)
    model.fit(X_train, y_train)
    save_model(model, f"models/{name}.pkl.bz2")


def train_model_nn(model, data, name, check, epoch, batch_size):
    """
    Function train neural-network model, fit it and save to file.

    :param model: <NNModel> neural-network model
    :param data: <pd.DataFrame> covtype.data
    :param name: <string> name of file, model should be saved
    :param check: <bool> check if file with model exists. If exists, return
    :param epoch: <int> model fitting parameter
    :param batch_size: <int> batch_size for fitting model
    :return: None
    """
    if check:
        if os.path.isfile(f"models/{name}.pkl.bz2"):
            return
    X_train, X_test, y_train, y_test = data_split(data)

    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)

    model.fit(X_train, y_train, class_weights, epoch, batch_size)

    save_data(model.history, "history_nn")
    training_curves(model.history)

    save_model(model, f"models/{name}.pkl.bz2")


def rescale_data(X):
    """
    Normalize data (min-max)

    :param X: <pd.DataFrame> data (e.g. covtype.data features).
    :return: <pd.DataFrame> scaled data
    """
    return (X - X.min()) / (X.max() - X.min())


def training_curves(history):
    """
    Function plot and save training curves of neural-network model to figures dir.

    :param history: <History> history object from tensorflow
    :return: None
    """
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle("Training curves - nn model")

    ax[0, 0].plot(history.history["loss"])
    ax[0, 0].plot(history.history["val_loss"])
    ax[0, 0].set(title="Training curves - loss", xlabel="epoch", ylabel="loss")
    ax[0, 0].legend(["Train", "Validation"])

    ax[0, 1].plot(history.history["accuracy"])
    ax[0, 1].plot(history.history["val_accuracy"])
    ax[0, 1].set(title="Training curves - accuracy", xlabel="epoch", ylabel="accuracy", ylim=[0, 1])
    ax[0, 1].legend(["Train", "Validation"])

    ax[1, 0].plot(history.history["f1_score"])
    ax[1, 0].plot(history.history["val_f1_score"])
    ax[1, 0].set(title="Training curves - f1_score", xlabel="epoch", ylabel="f1_score", ylim=[0, 1])
    ax[1, 0].legend(["Train", "Validation"])

    plt.savefig("figures/training_curves.png")
    plt.close()


def transform_parameters(parameters):
    """
    Function transform int-coded hyperparameters to parameters for neural-network model.
    Genotype -> Phenotype

    :param parameters: <list> list of ints valid to use with that function
    :return: parameters to create neural-network model.
    """
    activation_list = ['sigmoid', 'tanh', 'relu', 'elu']
    optimizer_list = ['adam', 'rmsprop', 'adamax', 'nadam']

    return parameters[0]/10, 2**parameters[1], parameters[2], activation_list[parameters[3]], \
        optimizer_list[parameters[4]], parameters[5]/10, 2**parameters[6], activation_list[parameters[7]]


def predict(name, X):
    """
    Load model and make predict on X data

    :param name: 'heuristic'|'hgb'|'nn'|'random_forrest' name of model
    :param X: <pd.DataFrame> features which should be used to predict data
    :return: <numpy.ndarray> vector of predicted values
    """
    if name in ["heuristic", "hgb", "nn", "random_forrest"]:
        ifile = bz2.BZ2File(f"models/{name}.pkl.bz2","rb")
        warnings.filterwarnings("ignore")
        model = pickle.load(ifile)
        warnings.filterwarnings("default")
        ifile.close()
    else:
        raise ValueError("name should be in ['heuristic', 'hgb', 'nn', 'random_forrest']")
    return model.predict(X)


def evaluate(name, data):
    """
    Function evaluate model. Used metrics - accuracy, f1_score_macro.
    Function also return confusion matrix

    :param name: 'heuristic'|'hgb'|'nn'|'random_forrest' name of model
    :param data: <pd.DataFrame> covtype.data
    :return: <tuple> tuple consist of dict with metrics and confusion matrix.
    """
    X_train, X_test, y_train, y_test = data_split(data)
    y_pred = predict(name, X_test)
    return ({"accuracy": accuracy_score(y_test, y_pred),
            "f1_score_macro": f1_score(y_test, y_pred, average="macro")},
            confusion_matrix(y_test, y_pred, labels=sorted(y_test.unique())))


def visualize(models, data):
    """
    Function compare models by ploting metrics, plot is saved to figures dir.

    :param models: <list> name of models.  'heuristic'|'hgb'|'nn'|'random_forrest'
    :param data: <pd.DataFrame> covtype.data
    :return: None
    """

    score = []
    confusion_matrices = []
    for model in models:
        result = evaluate(model, data)
        score.append(result[0])
        confusion_matrices.append(result[1])

    fig, ax = plt.subplots(2, 2, figsize=(10, 10))

    fig.suptitle("Comparison of models")

    barWidth = 0.2
    br = []

    for i in range(len(score)):
        if i == 0:
            br = np.arange(len(score[i]))
        else:
            br = [x + barWidth for x in br]
        ax[0, 0].bar(br, score[i].values(), width=barWidth, edgecolor='grey', label=models[i])

    ax[0, 0].set(title="Models score", xlabel="metric", ylabel="value", ylim=[0, 1])
    ax[0, 0].set_xticks([r + (len(models)-1)/2 * barWidth for r in range(len(score[0].values()))],score[0].keys())
    ax[0, 0].legend(models)

    axes = [ax[0, 1], ax[1, 0], ax[1, 1]]
    for index, ax in enumerate(axes):
        sns.heatmap(confusion_matrices[index+1], fmt='d', annot=True, square=True, cmap='gray_r', vmin=0, vmax=0,
                    linewidths=0.5, linecolor='k', cbar=False, ax=ax,
                    xticklabels=sorted(data.iloc[:, -1].unique()), yticklabels=sorted(data.iloc[:, -1].unique()))
        ax.set(title=f"{models[index+1]} confusion matrix", ylabel="true class", xlabel="predicted class")
    sns.despine(left=False, right=False, top=False, bottom=False)

    plt.savefig("figures/compare_models.png")