# Dependencies
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter('ignore', UserWarning)
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import sys
import argparse
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import itertools
from scipy import stats
from sklearn.metrics import auc, accuracy_score, roc_curve, precision_score, recall_score, f1_score, roc_auc_score
from lightgbm import LGBMClassifier
import lightgbm as lgb
import matplotlib.gridspec as gridspec
import seaborn as sns
import pylab as plot
import pandas
import time
from statsmodels.stats.outliers_influence import variance_inflation_factor    
from joblib import Parallel, delayed

# Function to calculate Variacne Inflation Factor for Pandas dataframe
def calculate_vif_(X):
    variables = [X.columns[i] for i in range(X.shape[1])]
    dropped=True
    while dropped:
        dropped=False
        print(len(variables))
        vif = Parallel(n_jobs=1,verbose=5)(delayed(variance_inflation_factor)(X[variables].values, ix) for ix in range(len(variables)))
        print(vif)
        maxloc = vif.index(max(vif))
        if max(vif) > thresh:
            print(time.ctime() + ' dropping \'' + X[variables].columns[maxloc] + '\' at index: ' + str(maxloc))
            variables.pop(maxloc)
            dropped=True

    print('Remaining variables:')
    print([variables])
    return X[[i for i in variables]]

def parse_args():
    parser = argparse.ArgumentParser(description = "", epilog = "")
    parser.add_argument("-df", "--dataFolder", help="Path to where the training data (TCGA, DepMap, Embedding) is stored (REQUIRED).", dest="dataFolder")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    available_samples = ["s1"]
    cancer_type_list = ["liver","breast","bladder", "colon", "ovarian", "kidney", "leukemia","pancreatic","lung"]
    orderFeatures = ["essentiality","mutation","expression", "e0", "e1", "e2", "e3", "e4", "e5", "e6", "e7", "e8", "e9", "e10", "e11", "e12", "e13", "e14", "e15", "e16", "e17", "e18", "e19", "e20", "e21", "e22", "e23", "e24", "e25", "e26", "e27", "e28", "e29", "e30", "e31", "label"]
    features = ["essentiality","mutation","expression", "e0", "e1", "e2", "e3", "e4", "e5", "e6", "e7", "e8", "e9", "e10", "e11", "e12", "e13", "e14", "e15", "e16", "e17", "e18", "e19", "e20", "e21", "e22", "e23", "e24", "e25", "e26", "e27", "e28", "e29", "e30", "e31"]

    for cancer_type in cancer_type_list:
        for inx, sampleNumber in enumerate(available_samples):
            # Load dataset
            data = pandas.read_csv(args.dataFolder + cancer_type.capitalize() + "/" + cancer_type + "_training_data_" + sampleNumber + ".dat", header=0, sep=",")
            data.drop("gene", axis=1, inplace=True)
            data = data[data['label'] != 2]

            dataframePositive = data[data['label'] == 1]
            dataframeNegative = data[data['label'] == 0]
            positiveSize = dataframePositive.shape[0]
            negativeSize = dataframeNegative.shape[0]

            # Set them the same size
            if(positiveSize > negativeSize):
                dataframePositive = dataframePositive.head(-(positiveSize-negativeSize))
            elif(negativeSize > positiveSize):
                dataframeNegative = dataframeNegative.head(-(negativeSize-positiveSize))

            data = dataframePositive.copy()
            data = pd.concat([dataframePositive, dataframeNegative])

            categorical_feats = [
                f for f in data.columns if data[f].dtype == 'object'
            ]

            categorical_feats
            for f_ in categorical_feats:
                data[f_], _ = pandas.factorize(data[f_])
                # Set feature type as categorical
                data[f_] = data[f_].astype('category')

            data = data.reindex(columns=orderFeatures)

            X = data[features] # Selecting your data

            vif = pd.DataFrame()
            vif["Feature"] = X.columns
            vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
            vif = vif.round(2)
            print("===")
            print(cancer_type)
            print(vif)
            print("===")
            vif.to_csv('output/feature_multicollinerity/' + cancer_type + '_multicollinearity_check.csv', index=False)