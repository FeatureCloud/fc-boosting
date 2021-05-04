import numbers
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils import Bunch
from redis_util import redis_set, redis_get


def set_X_y(df, label_col=None, return_X_y=False):
    """
    Parameters
    ----------
    df: data
    label_col: column name of label colun
    return_X_y: boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object. See
        below for more information about the `data` and `target` object.

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification labels

    (data, target) : tuple if ``return_X_y`` is True

    """
    if label_col is None:
        y = df.iloc[:, 0]
        X = df.drop(df.columns[[0]], axis=1)
    else:
        y = df.loc[:, label_col]
        X = df.drop(label_col, axis=1)

    y = np.array(y)
    X = np.array(X)

    if return_X_y:
        return X, y

    print(X)
    print(y)

    return Bunch(data=X, target=y)


def build_model(x_train, x_test, y_train, y_test):
    lr = float(redis_get("learning_rate"))
    if redis_get("random_state"):
        random_state = redis_get("random_state")
    else:
        random_state = None
    # Create adaboost object
    obj = AdaBoostClassifier(n_estimators=redis_get("n_estimators"), learning_rate=lr, random_state=random_state)
    # Train Adaboost 
    model = obj.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    if "acc" in redis_get("metric"):
        score = accuracy_score(y_test, y_pred)
    elif "matth" in redis_get("metric"):
        score = matthews_corrcoef(y_test, y_pred)
    elif "roc" in redis_get("metric") or "auc" in redic_get("metric"):
        score = roc_auc_score(y_test, y_pred)
    else:
        score = accuracy_score(y_test, y_pred)
    return score, model
