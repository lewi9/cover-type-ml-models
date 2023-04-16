import urllib.request
import os.path
import pickle
import bz2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split



def get_data():
    path = "data/covtype.data.gz"
    if not os.path.isfile(path):
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
        urllib.request.urlretrieve(url, path)
    return pd.read_csv(path, header = None)

def data_split(data):
    X = data.iloc[:,:-1]
    y = data.iloc[:,-1]
    return train_test_split(X, y, test_size=0.2, random_state=42,
                            stratify=y, shuffle = True)

def save_model(model, path):
    ofile = bz2.BZ2File(path,'wb')
    pickle.dump(model, ofile)
    ofile.close()

def train_model(model, data, name, check):
    if check:
        if os.path.isfile(f"models/{name}.pkl.bz2"):
            return
    np.random.seed(42)
    X_train, X_test, y_train, y_test = data_split(data)
    model.fit(X_train, y_train)
    print(model.score(X_test,y_test))
    save_model(model, f"models/{name}.pkl.bz2")

def rescale_data(X):
    return (X-X.min())/(X.max()-X.min())
    
