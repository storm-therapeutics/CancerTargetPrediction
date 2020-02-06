# Dependencies
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter('ignore', UserWarning)
import numpy as np
import pandas
import pandas as pd
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers.normalization import BatchNormalization
from keras.models import load_model
from keras import callbacks
import sys
from matplotlib import pyplot as plt
import itertools
import pickle
import argparse
from Utils import *

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

def parse_args():
    parser = argparse.ArgumentParser(description = "", epilog = "")
    parser.add_argument("-df", "--dataFolder", help="Path to where the training data (TCGA, DepMap, Embedding) is stored (REQUIRED).", dest="dataFolder")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    cancerTypes = ["bladder", "breast", "colon", "kidney", "leukemia", "liver", "lung", "ovarian", "pancreatic"]
    modelPerformancesFolder = "output/model_performance/"
    available_samples = ["s1","s2","s3","s4","s5","s6","s7","s8","s9","s10"]

    for cancerType in cancerTypes:
        gene_predictions = dict()
        models, model_name, genesTrainingSet = select_best_method(modelPerformancesFolder, args.dataFile, cancerType, available_samples, True)

        data_to_predict = getDataForPrediction(args.dataFile, cancerType, available_samples, genesTrainingSet)

        for model in models:
            geneNames = data_to_predict[0][0]

            # Do the predictions for each model
            classesProb = None
            if(model_name == "nn"):
                classesProb = model.predict(data_to_predict[dataIndex][1])
            else:
                classesProb = model.predict_proba(data_to_predict[dataIndex][1])

            classesProb = list(classesProb)
            if(model_name == "nn"):
                classesProb = [x[0] for x in classesProb]
            else:
                classesProb = [x[1] for x in classesProb]

            # Save the probabilities to the dict
            for predidx, geneName in enumerate(geneNames):
                if geneName not in gene_predictions:
                    gene_predictions[geneName] = list()

                gene_predictions[geneName].append(classesProb[predidx])

        # Aggregate the prediction probabilities
        for geneNameKey in gene_predictions:
            gene_predictions[geneNameKey] = np.median(gene_predictions[geneNameKey])

        # Get predictions
        listOfGeneTargetProbabilities = list()
        for geneNameKey in gene_predictions:
            listOfGeneTargetProbabilities.append((geneNameKey, gene_predictions[geneNameKey]))

        # Show the list
        listOfGeneTargetProbabilities.sort(key=lambda x: x[1], reverse=True)
        outF = open("predictions_" + str(cancerType) + ".csv", "w")
        outF.write("symbol,probability\n")
        for line in listOfGeneTargetProbabilities:
            outF.write(str(line[0]) + "," + str(line[1]))
            outF.write("\n")
        outF.close()