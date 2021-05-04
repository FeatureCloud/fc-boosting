import json
import jsonpickle
import numpy as np
import os
import pandas as pd
import pickle
from flask import current_app
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import train_test_split

from redis_util import redis_set, redis_get
from .helpfunctions import build_model, set_X_y


def read_input(input_dir: str):
    """
    Reads all files stored in 'input_dir'.
    :param input_dir: The input directory containing the files.
    :return: None
    """
    data = None
    filename = redis_get('input_filename')
    missing_data = redis_get('missing_data')
    try:
        current_app.logger.info('[API] Parsing data of ' + input_dir)
        current_app.logger.info('[API] ' + filename)
        if filename.endswith(".csv"):
            sep = ','
            data = pd.read_csv(input_dir + '/' + filename, sep=sep)
        elif filename.endswith(".tsv"):
            sep = '\t'
            data = pd.read_csv(input_dir + '/' + filename, sep=sep)
        if missing_data == "mean":
            data.fillna(data.mean(), inplace=True)
        elif missing_data == "median":
            data.fillna(data.median(), inplace=True)
        elif missing_data == "drop":
            data.dropna(inplace=True)

        current_app.logger.info('[API] ' + str(data))

        return data

    except Exception as e:
        current_app.logger.info('[API] could not read files', e)


def calculate_global_model():
    """
    Combines the models of all clients in a list.
    :return: None
    """
    current_app.logger.info('[API] Combine all models')
    global_data = redis_get('global_data')
    global_model = jsonpickle.decode(global_data[0])
    for model in global_data[1:]:
        global_model.estimators_ = global_model.estimators_ + jsonpickle.decode(model).estimators_
    redis_set('global_model', jsonpickle.encode(global_model))


def calculate_local_model():
    """
    Perform local boosting
    :return: the model
    """
    current_app.logger.info('[API] Perform local boosting')
    d = redis_get('files')

    if d is None:
        current_app.logger.info('[API] No data available')
        return None
    else:
        client_id = redis_get('id')

        df = set_X_y(d, label_col=redis_get("label_col"))

        # Split dataset into training set and test set
        # 70% training and 30% test
        x_train, x_test, y_train, y_test = train_test_split(df.get("data"), df.get("target"),
                                                            test_size=redis_get("test_size"),
                                                            stratify=df.get("target"),
                                                            random_state=redis_get("random_state"))
        redis_set("test_set", jsonpickle.encode([x_test, y_test]))
        score, model = build_model(x_train, x_test, y_train, y_test)
        saved_model = jsonpickle.encode(model)
        metric = redis_get("metric")
        current_app.logger.info(f'[API] Local AdaBoost classifier model {metric} {client_id}: {score} ')
        redis_set('score_single', score)
        return saved_model


def calculate_average():
    global_model = jsonpickle.decode(redis_get('global_model'))
    client_id = redis_get('id')
    test_set = jsonpickle.decode(redis_get("test_set"))
    x_test = test_set[0]
    y_test = test_set[1]

    clf = global_model
    sum_pred = clf.predict(x_test)

    if "acc" in redis_get("metric"):
        score = accuracy_score(y_test, sum_pred)
    elif "matth" in redis_get("metric"):
        score = matthews_corrcoef(y_test, sum_pred)
    elif "roc" in redis_get("metric") or "auc" in redis_get("metric"):
        score = roc_auc_score(y_test, sum_pred)
    else:
        score = accuracy_score(y_test, sum_pred)
    current_app.logger.info(
        f'[API] Combined AdaBoost classifier model score on local test data for {client_id}: {score}, Predictions: {sum_pred}')
    redis_set('predictions', sum_pred)
    redis_set('score_combined', score)

    return score


def write_results(output_dir: str, model=None, score=None, plot=None):
    """
    Writes the results of global_km to the output_directory.
    :param results: Global results calculated from the local counts of the clients
    :param output_dir: String of the output directory. Usually /mnt/output
    :return: None
    """
    current_app.logger.info("[API] Write results to output folder")

    if model is not None:
        # save the model to disk
        filename = output_dir + '/' + 'global_boosting_classifier.sav'
        pickle.dump(model, open(filename, 'wb'))
    if score is not None:
        filename = output_dir + '/eval_on_local_testset.csv'
        score_df = pd.DataFrame(index=["local_model", "global_model"], columns=[redis_get("metric")],
                                data=[redis_get("score_single"), redis_get("score_combined")])
        score_df.to_csv(filename)
    if plot is not None:
        filename = 'plot.png'
        plot.savefig(filename)
