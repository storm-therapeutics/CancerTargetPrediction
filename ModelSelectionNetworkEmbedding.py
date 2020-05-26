# Dependencies
import warnings
warnings.filterwarnings("ignore")

import random
random.seed(9999)

import argparse
import pickle
import numpy as np
import pandas
import sys
import itertools
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import auc, accuracy_score, confusion_matrix, roc_curve, precision_score, recall_score, f1_score
from Utils import *

# Keras+Tensorflow libraries
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adadelta
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers.normalization import BatchNormalization
from keras import callbacks



# Config Keras
num_cores = 8
GPU = True
CPU = False

if GPU:
    num_GPU = 1
    num_CPU = 1
if CPU:
    num_CPU = 1
    num_GPU = 0

config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
session = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=config)
tf.compat.v1.keras.backend.set_session(session)

# AUC ROC metric for Keras
def auc_roc(y_true, y_pred):
    # any tensorflow metric
    value, update_op = tf.contrib.metrics.streaming_auc(y_pred, y_true)

    # find all variables created for this metric
    metric_vars = [i for i in tf.local_variables() if 'auc_roc' in i.name.split('/')[1]]

    # Add metric variables to GLOBAL_VARIABLES collection.
    # They will be initialized for new session.
    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

    # force to update metric values
    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
        return value

def trainRFmodel(X, Y):
    # Get the training and test data
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

    # CV
    grid_search_cv = GridSearchCV(
        estimator = RandomForestClassifier(
            n_jobs = -1
        ),
        param_grid = {
            'max_depth' : [4, 5, 6, 7, 8] ,
            'n_estimators': [300, 500, 700, 850, 1000], 
            'max_features': ['sqrt', 'log2', 0.3, 0.5], 
        },    
        scoring = "accuracy",
        cv = StratifiedKFold(
            n_splits=5,
            shuffle=True
        ),
        n_jobs = -1,  
        verbose = 1,
        refit = True
    )

    # Fit the model
    model = grid_search_cv.fit(X_train, y_train)

    y_pred = model.best_estimator_.predict(X_test)
    y_pred_rt = model.best_estimator_.predict_proba(X_test)[:, 1]

    accuracy = str(accuracy_score(y_test, y_pred))
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_rt)
    auc_value = str(auc(fpr, tpr))
    precision = str(precision_score(y_test, y_pred))
    recall = str(recall_score(y_test, y_pred))
    f1score = str(f1_score(y_test, y_pred, average="weighted"))

    return [accuracy, auc_value, precision, recall, f1score, y_test, y_pred, y_pred_rt, model.best_estimator_]

def trainSVMmodel(X, Y):
    # Get the training and test data
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

    # CV
    grid_search_cv = GridSearchCV(
        estimator = SVC(
            probability = True
        ),
        param_grid = [{'kernel':['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
                      {'kernel':['linear'], 'C': [1, 10, 100, 1000]} ],
        scoring = "accuracy",
        cv = StratifiedKFold(
            n_splits=5,
            shuffle=True
        ),
        n_jobs = -1, 
        verbose = 1,
        refit = True
    )

    # Fit the model
    model = grid_search_cv.fit(X_train, y_train)

    y_pred = model.best_estimator_.predict(X_test)
    y_pred_rt = model.best_estimator_.predict_proba(X_test)[:, 1]

    accuracy = str(accuracy_score(y_test, y_pred))
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_rt)
    auc_value = str(auc(fpr, tpr))
    precision = str(precision_score(y_test, y_pred))
    recall = str(recall_score(y_test, y_pred))
    f1score = str(f1_score(y_test, y_pred, average="weighted"))

    return [accuracy, auc_value, precision, recall, f1score, y_test, y_pred, y_pred_rt, model.best_estimator_]

def trainXGBoostmodel(X, Y):
    # Get the training and test data
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

    # CV
    grid_search_cv = GridSearchCV(
        estimator = xgb.XGBClassifier(
           # n_jobs = -1,
            objective = 'binary:logistic',
            eval_metric = "auc",
            silent=1,
            tree_method='approx'
        ),
        param_grid = {
            'max_features': ['sqrt', 'log2', 0.3, 0.5], 
            'learning_rate': (0.05, 0.1),
            'max_depth' : [4, 5, 6, 7, 8] ,
            'n_estimators': [300, 500, 700, 850, 1000], 
        },    
        scoring = "accuracy",
        cv = StratifiedKFold(
            n_splits=5,
            shuffle=True
        ),
        n_jobs = -1, 
        verbose = 1,
        refit = True
    )

    # Fit the model
    model = grid_search_cv.fit(X_train, y_train)

    y_pred = model.best_estimator_.predict(X_test)
    y_pred_rt = model.best_estimator_.predict_proba(X_test)[:, 1]

    accuracy = str(accuracy_score(y_test, y_pred))
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_rt)
    auc_value = str(auc(fpr, tpr))
    precision = str(precision_score(y_test, y_pred))
    recall = str(recall_score(y_test, y_pred))
    f1score = str(f1_score(y_test, y_pred, average="weighted"))

    return [accuracy, auc_value, precision, recall, f1score, y_test, y_pred, y_pred_rt, model.best_estimator_]

def trainNNmodel(X, Y):
    # Get the training and test data
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

    # Function to create the NN model, required for the wrapper
    def create_keras_model():
        model = Sequential()
        model.add(Dense(64, input_dim=X.shape[1], kernel_initializer='glorot_normal', activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(128, kernel_initializer='glorot_normal', activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(16, kernel_initializer='glorot_normal', activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])
        return model

    # Fit the model
    early_stop = callbacks.EarlyStopping(monitor="accuracy", patience=50, mode='max') 
    callbacks_list = [early_stop]

    estimator = KerasClassifier(build_fn=create_keras_model, epochs=200, batch_size=12, verbose=0, callbacks=callbacks_list)
    estimator.fit(X_train, y_train, batch_size=12, epochs=200, verbose=1, callbacks=callbacks_list)

    y_pred = estimator.predict(X_test)
    y_pred = [item for sublist in y_pred for item in sublist]
    y_pred_rt = estimator.predict_proba(X_test)[:, 1]

    accuracy = str(accuracy_score(y_test, y_pred))
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_rt)
    auc_value = str(auc(fpr, tpr))
    precision = str(precision_score(y_test, y_pred))
    recall = str(recall_score(y_test, y_pred))
    f1score = str(f1_score(y_test, y_pred, average="weighted"))

    return [accuracy, auc_value, precision, recall, f1score, y_test, y_pred, y_pred_rt, estimator.model]

def trainLRmodel(X, Y):
    # Get the training and test data
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

    # Fit the model
    model = LogisticRegression().fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_rt = model.predict_proba(X_test)[:, 1]

    accuracy = str(accuracy_score(y_test, y_pred))
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_rt)
    auc_value = str(auc(fpr, tpr))
    precision = str(precision_score(y_test, y_pred))
    recall = str(recall_score(y_test, y_pred))
    f1score = str(f1_score(y_test, y_pred, average="weighted"))

    return [accuracy, auc_value, precision, recall, f1score, y_test, y_pred, y_pred_rt, model]

def parse_args():
    parser = argparse.ArgumentParser(description = "", epilog = "")
    parser.add_argument("-df", "--dataFolder", help="Path to where the training data (TCGA, DepMap, Embedding) is stored (REQUIRED).", dest="dataFolder")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    model_selection_statistics_output_path = "output/models_performance/"
    cancerTypes = ["bladder", "breast", "colon", "kidney", "leukemia", "liver", "lung", "ovarian", "pancreatic"] 
    available_samples = ["s1","s2","s3","s4","s5","s6","s7","s8","s9","s10"]
    model_list = ["lr", "svm", "xgboost", "nn", "rf"]

    for cancer_type in cancer_type_list:

        dataframes = getDatasetsNE(args.dataFolder, cancer_type, available_samples)

        # Structures to save the predictions and statistics of each of the ten models
        all_model_predictions = list()
        all_model_predictions.append("model,sampleid,y_test,y_pred,y_pred_rt")
        all_model_statistics = list()
        all_model_statistics.append("model,sampleid,accuracy,auc,precision,recall,f1score")
        all_model_objects = dict()

        for modelName in model_list:
            print("wtf")
            print(modelName)
            model_predictions = list()
            model_statistics = list()

            for inx, df in enumerate(dataframes):
                # Get the indepenendant and dependant variables
                X = df.values[:,0:df.values.shape[1]-1].astype(float)
                Y = df['label'].values

                # Train the corresponding model       
                if modelName == "rf":
                    model_results = trainRFmodel(X, Y)
                elif modelName == "lr":
                    model_results = trainLRmodel(X, Y)
                elif modelName == "svm":
                    model_results = trainSVMmodel(X, Y)
                elif modelName == "xgboost":
                    model_results = trainXGBoostmodel(X, Y)
                elif modelName == "nn":
                    model_results = trainNNmodel(X, Y)
                else:
                    print("ERROR: The given model does not exist, valid options: lr, rf, svm, xgboost, nn")
                    sys.exit(1)
                    
                # Save the results
                model_statistics.append(modelName + "," + str(inx+1) + "," + model_results[0] + "," + model_results[1] + "," + model_results[2] + "," + model_results[3] + "," + model_results[4])
                model_predictions.append(modelName + "," + str(inx+1) + "," + "|".join(map(str, model_results[5])) + "," + "|".join(map(str, model_results[6])) + "," + "|".join(map(str, model_results[7])))
                all_model_predictions.append(model_predictions[-1])
                all_model_statistics.append(model_statistics[-1])

                # Save the model
                if modelName not in all_model_objects:
                    all_model_objects[modelName] = list()

                all_model_objects[modelName].append(model_results[8])

            # Concatenate all predictions and calculate overall statistics
            y_test_concatenated = list()
            y_pred_concatenated = list()
            y_pred_rt_concatenated = list()

            for element in model_predictions:
                element_splitted = element.split(",")
                element_y_test = element_splitted[2].split("|")
                element_y_pred = element_splitted[3].split("|")
                element_y_pred_rt = element_splitted[4].split("|")

                y_test_concatenated = y_test_concatenated + element_y_test
                y_pred_concatenated = y_pred_concatenated + element_y_pred
                y_pred_rt_concatenated = y_pred_rt_concatenated + element_y_pred_rt

            y_pred_rt_concatenated = np.array(y_pred_rt_concatenated).astype(np.float)
            y_test_concatenated = np.array(y_test_concatenated).astype(np.float)
            y_pred_concatenated = np.array(y_pred_concatenated).astype(np.float)

            accuracy = str(accuracy_score(y_test_concatenated, y_pred_concatenated))
            fpr, tpr, thresholds = roc_curve(y_test_concatenated, y_pred_rt_concatenated)
            auc_value = str(auc(fpr, tpr))
            precision = str(precision_score(y_test_concatenated, y_pred_concatenated))
            recall = str(recall_score(y_test_concatenated, y_pred_concatenated))
            f1score = str(f1_score(y_test_concatenated, y_pred_concatenated, average="weighted"))
            
            all_model_statistics.append(modelName + ",overall," + accuracy + "," + auc_value + "," + precision + "," + recall + "," + f1score)

        # Save statistics to file
        outputFileName = obtainOutputFileName(model_selection_statistics_output_path, cancer_type)
        writeToFile(outputFileName, all_model_statistics)

        # Save predictions to file
        outputFileName = obtainPredictionsOutputFileName(model_selection_statistics_output_path, cancer_type)
        writeToFile(outputFileName, all_model_predictions)

        # Save models to pickle object
        for key in all_model_objects:
            for mldInx, savedModel in enumerate(all_model_objects[key]):
                pathToSave = obtainOutputModelFileName(model_selection_statistics_output_path, cancer_type, key, str(mldInx+1))
                with open(pathToSave, 'wb') as f:
                    pickle.dump(savedModel, f)