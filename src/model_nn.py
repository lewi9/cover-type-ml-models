import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.core.callback import Callback
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.optimize import minimize
from sklearn.utils import class_weight
from src.utils import rescale_data, data_split, save_data


class NNModel:

    def __init__(self, dropout=0.5, units=64, layers=2, activation='relu', optimizer='adam', dropout_f=0.5,
                 units_f=64, activation_f='relu', features=54, classes=7):
        """
        init NNModel. It uses tensorflow API to build itself.

        :param dropout: <float> dropout value of hidden layers
        :param units: <int> number of units in hidden layers
        :param layers: <int> number of hidden layers
        :param activation: <string> activation function of hidden layers
        :param optimizer: <string> optimizer of model
        :param dropout_f: <float> dropout of the first layer
        :param units_f: <int> number of units in the first layer
        :param activation_f: <string> activation function of hidden layer
        :param features: <int> number of features
        :param classes: <int> number of classes
        """

        self.history = None
        f1score = tfa.metrics.F1Score(num_classes=classes, average='macro')

        self._model = tf.keras.models.Sequential()

        self._model.add(tf.keras.layers.Dense(units_f, input_shape=(features,), activation=activation_f))
        self._model.add(tf.keras.layers.Dropout(dropout_f))

        for _ in range(layers):
            self._model.add(tf.keras.layers.Dense(units, activation=activation))
            self._model.add(tf.keras.layers.Dropout(dropout))

        self._model.add(tf.keras.layers.Dense(classes, activation='softmax'))

        self._model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy', f1score])

    def fit(self, X, y, class_weights, epochs=20, batch_size=128):
        """
        Method fit model.

        :param X: <pd.DataFrame> feautures (train data)
        :param y: <pd.DataFrame> classes (train data)
        :param class_weights: <list> of class_weights
        :param epochs: <int> number of epochs to train model
        :param batch_size: <int> batch_size used to train model
        :return: <NNModel>
        """
        X = rescale_data(X)
        X_train, X_test, y_train, y_test = data_split(pd.concat([X, y], axis=1))
        y_train = pd.get_dummies(y_train)
        y_test = pd.get_dummies(y_test)

        class_weights_dict = {}
        for index, weight in enumerate(class_weights):
            class_weights_dict[index] = weight

        self.history = self._model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                                       validation_data=(X_test, y_test), class_weight=class_weights_dict)
        return self

    def predict(self, X, batch_size=128):
        """
        Method predict values

        :param X: <pd.DataFrame> feautures
        :param batch_size: <int> batch_size used to make predict
        :return: <list> predicted classes
        """
        X = rescale_data(X)

        return [i+1 for i in np.argmax(self._model.predict(X, batch_size=batch_size), axis=1)]

    def score(self, X, y, batch_size):
        """
        Method evaluate model

        :param X: <pd.DataFrame> feautures
        :param y: <pd.DataFrame> classes
        :param batch_size: <int> batch_size used to evaluation
        :return: <list> value of loss function and metrics (accuracy, f1_score)
        """
        y = pd.get_dummies(y)
        X = rescale_data(X)

        return self._model.evaluate(X, y, batch_size)

    @staticmethod
    def tuning_nn(data, pop=6, gen=6, epoch=6):
        """
        Method search hyperparameters to model.
        It uses NSGA-II non-dominated sortic genetic algorithm.

        :param data: <pd.DataFrame> with data
        :param pop: <int> population per generation
        :param gen: <int> number of generations
        :param epoch: <int> number of epoch used to train model
        :return: <list> list of int-coded hyperparameters
        """

        # Cut data for training models.
        data_cutted = data.sample(frac=0.4, random_state=42)
        X_train, X_test, y_train, y_test = data_split(data_cutted)

        # Compute class_weights
        class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)

        class MyProblem(Problem):

            def __init__(self):
                """
                Init a Problem (this is object that represents optimization problem).
                n_var - number of variables in solution vector
                n_obj - number of function to optimize (here there are loss, 1-accuracy, 1-f1_score)
                xl - lower limits for n_var
                xu - upper limist for n_var
                vtype - type of optimized variables (type hint for user)
                """
                super().__init__(n_var=9, n_obj=3, xl=[2, 4, 3, 2, 0, 0, 4, 3, 0],
                                 xu=[13, 9, 8, 4, 3, 3, 9, 10, 3], vtype=int)
                self.activation = ['sigmoid', 'tanh', 'relu', 'elu']
                self.optimizer = ['adam', 'rmsprop', 'adamax', 'nadam']

            def validate(self, x):
                """
                Check vector (solution) that is valid to create NNModel
                :param x: <list> vector of integers
                :return: <list> valid vector of integers
                """
                if x[1] < 0 or x[1] >= 10:
                    x[1] = 5
                if x[4] < 0 or x[4] >= len(self.activation):
                    x[4] = 0
                if x[5] < 0 or x[5] >= len(self.optimizer):
                    x[5] = 0
                if x[6] < 0 or x[6] >= 10:
                    x[6] = 5
                if x[8] < 0 or x[8] >= len(self.activation):
                    x[8] = 0
                x = [abs(elem) for elem in x]
                return x

            def _evaluate(self, in_vec, out, *args, **kwargs):
                """
                There is representation of problem that should be optimized by algorithm.

                :param in_vec: <list> list of list. Every list represents one solution (member of population)
                :param out: <list> list of list. Every list represents calculated fitness function for
                            n(=3) functions (loss, accuracy, f1_score)
                :param args: default
                :param kwargs: default
                :return: None
                """
                out_vec = []

                # x[0] - batch_size - [4,8,16,32,64,128,256,512,1024,2048,4096,8192]
                # x[1] - dropout - validate [0.4,0.5,0.6,0.7,0.8,0.9]
                # x[2] - units [8,16,32,64,128,256]
                # x[3] - layers [2,3,4]
                # x[4] - activation - validate - ['sigmoid', 'tanh', 'relu', 'elu']
                # x[5] - optimizer - validate - ['adam', 'rmsprop', 'adamax', 'nadam']
                # x[6] - dropout_first_layer - validate - [0.4,0.5,0.6,0.7,0.8,0.9]
                # x[7] - units_first_layer - [8,16,32,64,128,256,512,1024]
                # x[8] - activate_first_layer - validate - ['sigmoid', 'tanh', 'relu', 'elu']

                for x in in_vec:
                    x = self.validate(x)

                    # Create model. Should be used transform_parameters from utils.
                    # Genotype -> Phenotype
                    model = NNModel(x[1]/10, 2**x[2], x[3], self.activation[x[4]], self.optimizer[x[5]],
                                    x[6]/10, 2**x[7], self.activation[x[8]], len(X_train.columns),
                                    len(y_train.unique()))


                    # Fit model
                    model.fit(X_train, y_train, class_weights, epoch, 2**x[0])

                    # evaluate model - calc fitness function
                    loss, accuracy, f1score = model.score(X_test, y_test, 2 ** x[0])
                    if f1score < 0.1:
                        f1score = model.history.history["val_f1_score"][-1]
                    if loss != loss:
                        loss = model.history.history["val_loss"][-1]
                    out_vec.append([loss, 1 - accuracy, 1 - f1score])

                # Fitness function
                out["F"] = out_vec.copy()

        class MyCallback(Callback):
            def __init__(self) -> None:
                """
                Callback for optimize algorithm to not lost result if something went wrong (e.g. computer crashed)
                """
                super().__init__()
                self.pareto_front = None
                self.pareto_set = None

            def notify(self, algorithm):
                """
                Method that specify callback behaviour.

                :param algorithm: default
                :return: None
                """
                self.pareto_front = algorithm.opt.get("F")
                self.pareto_set = algorithm.opt.get("X")

                # Save data to files.
                save_data(self.pareto_set, "hyperparameters_set")
                save_data(self.pareto_front, "result_of_hyperparameters")

        # Define problem to optimize
        problem = MyProblem()

        # Define algorithm of searching optimum
        method = NSGA2(pop_size=pop,
                       sampling=IntegerRandomSampling(),
                       crossover=SBX(prob=0.8, eta=5, vtype=float, repair=RoundingRepair()),
                       mutation=PM(prob=1.0, eta=15, vtype=float, repair=RoundingRepair()),
                       eliminate_duplicates=True,
                       )

        # Optimize problem
        res = minimize(problem, method, termination=('n_gen', gen), seed=42, save_history=True,
                       return_least_infeasible=True, callback=MyCallback())
        # Return result
        return res
