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

def display_distributions(actual_imp_df_, null_imp_df_, feature_):
    plt.figure(figsize=(13, 6))
    gs = gridspec.GridSpec(1, 1)
    ax = plt.subplot(gs[0, 0])
    fig = plt.gcf()
    fig.set_size_inches(8, 4)
    params = {'legend.fontsize': 14, 'legend.handlelength': 2}
    plot.rcParams.update(params)
    null_imp_df_.loc[null_imp_df_['feature'] == feature_, 'importance_gain'].plot.kde(ax=ax, legend=True, label='Null distribution')
    plt.axvline(actual_imp_df_.loc[actual_imp_df_['feature'] == feature_, 'importance_gain'].mean(), 0, np.max(null_imp_df_.loc[null_imp_df_['feature'] == feature_, 'importance_gain'].values), color='r', label='Observed importance')
    ax.legend(loc=1)
    plt.xlabel('Importance score', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.tight_layout()
    plt.savefig(feature_ + "_importance_plot.svg")
    plt.savefig(feature_ + "_importance_plot.png")
    plt.show()

def parse_args():
    parser = argparse.ArgumentParser(description = "", epilog = "")
    parser.add_argument("-df", "--dataFolder", help="Path to where the training data (TCGA, DepMap, Embedding) is stored (REQUIRED).", dest="dataFolder")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    available_samples = ["s1","s2","s3","s4","s5","s6","s7","s8","s9","s10"]
    cancer_type_list = ["liver","breast","bladder", "colon", "ovarian", "kidney", "leukemia","pancreatic","lung"]
    orderFeatures = ["essentiality","mutation","expression", "e0", "e1", "e2", "e3", "e4", "e5", "e6", "e7", "e8", "e9", "e10", "e11", "e12", "e13", "e14", "e15", "e16", "e17", "e18", "e19", "e20", "e21", "e22", "e23", "e24", "e25", "e26", "e27", "e28", "e29", "e30", "e31", "label"]

    for cancer_type in cancer_type_list:
        cancerCorr = pd.DataFrame()
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
            # data is the dataframe
            corr = data.corr(method="spearman")
            if inx == 0:
                cancerCorr = corr
            else:
                df_concat = pd.concat((cancerCorr, corr))
                by_row_index = df_concat.groupby(df_concat.index)
                df_mean = by_row_index.mean()
                cancerCorr = df_mean
                
        cancerCorr = cancerCorr.reindex(orderFeatures)
        # print(cancerCorr)
        print(cancer_type)

        cancerCorr.to_csv('output/feature_correlation/' + cancer_type + '_feature_correlation.csv')
        mask = np.zeros_like(cancerCorr)
        mask[np.triu_indices_from(mask)] = True
        
        plt.clf()
        plt.figure(figsize=(10,10))
        sns.heatmap(cancerCorr, mask=mask, 
        xticklabels=cancerCorr.columns.values, 
        yticklabels=cancerCorr.columns.values, 
        vmin=-1, 
        vmax=1, 
        cmap="coolwarm", 
        center=0)
        plt.savefig('output/feature_correlation/' + cancer_type + '_feature_correlation.png', dpi=300)
        plt.savefig('output/feature_correlation/' + cancer_type + '_feature_correlation.eps', dpi=300)
        # plt.show()