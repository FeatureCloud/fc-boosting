import json
import jsonpickle
import numpy as np
import os
import pandas as pd
import pickle
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import train_test_split

from .helpfunctions import build_model, set_X_y

def read_input(input_dir: str, self):
    """
    Reads all files stored in 'input_dir'.
    :param input_dir: The input directory containing the files.
    :return: None
    """
    data = None
    filename = self.load('input_filename')
    missing_data = self.load('missing_data')
    try:
        self.log('[API] Parsing data of ' + input_dir)
        self.log('[API] ' + filename)
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

        self.log('[API] ' + str(data))

        return data

    except Exception as e:
        self.log('[API] could not read files', e)


def calculate_global_model(self):
    """
    Combines the models of all clients in a list.
    :return: None
    """
    self.log('[API] Combine all models')
    global_data = self.load('global_data')
    global_model = global_data[0]

    for model in global_data[1:]:
        global_model.estimators_ = global_model.estimators_ + model.estimators_
    self.store('global_model', global_model)

def calculate_local_model(self):
    """
    Perform local boosting
    :return: the model
    """
    self.log('[API] Perform local boosting')
    d = self.load('files')

    if d is None:
        self.log('[API] No data available')
        return None
    else:

        df = set_X_y(d, label_col=self.load("label_col"))

        # Split dataset into training set and test set
        # 70% training and 30% test
        x_train, x_test, y_train, y_test = train_test_split(df.get("data"), df.get("target"),
                                                            test_size=self.load("test_size"),
                                                            stratify=df.get("target"),
                                                            random_state=self.load("random_state"))
        self.store("test_set", jsonpickle.encode([x_test, y_test]))
        score, model = build_model(x_train, x_test, y_train, y_test, self)
        metric = self.load("metric")
        self.log(f'[API] Local AdaBoost classifier model {metric} {self.id}: {score} ')

        self.store('score_single', score)
        return model


def calculate_average(self):
    global_model = self.load('global_model')
    test_set = jsonpickle.decode(self.load("test_set"))
    x_test = test_set[0]
    y_test = test_set[1]

    clf = global_model
    sum_pred = clf.predict(x_test)

    if "acc" in self.load("metric"):
        score = accuracy_score(y_test, sum_pred)
    elif "matth" in self.load("metric"):
        score = matthews_corrcoef(y_test, sum_pred)
    elif "roc" in self.load("metric") or "auc" in self.load("metric"):
        score = roc_auc_score(y_test, sum_pred)
    else:
        score = accuracy_score(y_test, sum_pred)
    self.log(
        f'[API] Combined AdaBoost classifier model score on local test data for {self.id}: {score}, Predictions: {sum_pred}')
    self.store('predictions', sum_pred)
    self.store('score_combined', score)

    return score


def write_results(self, output_dir: str, model=None, score=None, plot=None):
    """
    Writes the results of global_km to the output_directory.
    :param results: Global results calculated from the local counts of the clients
    :param output_dir: String of the output directory. Usually /mnt/output
    :return: None
    """
    self.log("[API] Write results to output folder")

    if model is not None:
        # save the model to disk
        filename = output_dir + '/' + 'global_boosting_classifier.sav'
        pickle.dump(model, open(filename, 'wb'))
    if score is not None:
        filename = output_dir + '/eval_on_local_testset.csv'
        score_df = pd.DataFrame(index=["local_model", "global_model"], columns=[self.load("metric")],
                                data=[self.load("score_single"), self.load("score_combined")])
        score_df.to_csv(filename)
        
    if plot is not None:
        filename = 'plot.png'
        plot.savefig(filename)
