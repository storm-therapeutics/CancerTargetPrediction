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

def get_feature_importances(data, shuffle, seed=None):
    train_features = [f for f in data if f not in ['label']]
    
    # Shuffle target if required
    y = data['label'].copy()
    if shuffle:
        y = data['label'].copy().sample(frac=1.0)

    # Fit LightGBM in Random Forest mode (quicker than sklearn RandomForest)
    dtrain = lgb.Dataset(data[train_features], y, free_raw_data=False, silent=True)
    lgb_params = {
        'objective': 'binary',
        'boosting_type': 'rf',
        'learning_rate': .01,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'num_leaves': 12,
        'max_depth': -1,
        'n_jobs': -1,
        'min_split_gain': .00001,
        'reg_alpha': .00001,
        'reg_lambda': .00001,
        'metric': 'auc',
        'bagging_freq': 1,
        'verbose': -1
    }
    
    # Fit the model
    clf = lgb.train(params=lgb_params, train_set=dtrain, num_boost_round=200, categorical_feature=categorical_feats)

    # Get feature importances
    imp_df = pandas.DataFrame()
    imp_df["feature"] = list(train_features)
    imp_df["importance_gain"] = clf.feature_importance(importance_type='gain')
    imp_df["importance_split"] = clf.feature_importance(importance_type='split')
    imp_df['trn_score'] = roc_auc_score(y, clf.predict(data[train_features]))
    
    return imp_df

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

    correlation_scores = []
    plotDistributions = True

    for cancer_type in cancer_type_list:
        for sampleNumber in available_samples:
            # Load dataset
            data = pandas.read_csv(args.dataFile + cancer_type.capitalize() + "/" + cancer_type + "_training_data_" + sampleNumber + ".dat", header=0, sep=",")
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

            msk = np.random.rand(len(data)) < 0.7
            traindf = data[msk].copy()
            testdf = data[~msk].copy()
            data = traindf

            # Build Null Importances distribution
            null_imp_df = pandas.DataFrame()
            nb_runs = 100
            for i in range(nb_runs):
                # Get current run importances
                imp_df = get_feature_importances(data=data, shuffle=True)
                imp_df['run'] = i + 1 
                null_imp_df = pandas.concat([null_imp_df, imp_df], axis=0)
            
            # Build actual importances distribution
            actual_imp_df = pandas.DataFrame()
            imp_df = get_feature_importances(data=data, shuffle=False)
            for i in range(nb_runs):
                imp_df['run'] = i + 1 
                # Concat the latest importances with the old ones
                actual_imp_df = pandas.concat([actual_imp_df, imp_df], axis=0)

            # display_distributions(actual_imp_df_=actual_imp_df, null_imp_df_=null_imp_df, feature_='mutation')
            # display_distributions(actual_imp_df_=actual_imp_df, null_imp_df_=null_imp_df, feature_='expression')
            # display_distributions(actual_imp_df_=actual_imp_df, null_imp_df_=null_imp_df, feature_='essentiality')

            null_imp_df.to_csv('null_importances_distribution_rf.csv')
            actual_imp_df.to_csv('actual_importances_ditribution_rf.csv')

            # Z-score calculation
            for _f in actual_imp_df['feature'].unique():
                f_null_imps = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_gain'].values
                f_act_imps = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_gain'].values
                importancesDistribution = f_null_imps.tolist()
                importancesDistribution.append(f_act_imps[0])
                zScores = stats.zscore(importancesDistribution)
                correlation_scores.append((_f, zScores[-1], sampleNumber))

        corr_scores_df = pandas.DataFrame(correlation_scores, columns=['feature', 'z_score', 'sample_number'])
        corr_scores_df = corr_scores_df.groupby(['feature', 'sample_number'], as_index=False).mean()
        corr_scores_df = corr_scores_df.groupby(['feature'], as_index=False).mean()
        corr_scores_df = corr_scores_df.sort_values('feature', ascending=True)
        corr_scores_df.to_csv("output/feature_importance/" + cancer_type + "_feature_importance.csv", encoding='utf-8', index=False)
        
        # Plot the distributions?
        if plotDistributions:
            fig = plt.figure(figsize=(16, 16))
            gs = gridspec.GridSpec(1, 1)

            # Plot Gain importances
            ax = plt.subplot(gs[0, 0])
            sns.barplot(x='gain_score', y='feature', data=corr_scores_df.sort_values('gain_score', ascending=False), ax=ax)
            ax.set_title('Feature scores wrt gain importances ', fontweight='bold', fontsize=14)

            rects = ax.patches

            # For each bar: Place a label
            for index, rect in enumerate(rects):
                # Get X and Y placement of label from rect.
                x_value = rect.get_width()
                y_value = rect.get_y() + rect.get_height() / 2

                # Number of points between bar and label
                space = 5
                
                # Vertical alignment for positive values
                ha = 'left'

                # If value of bar is negative: Place label left of bar
                if x_value < 0:
                    # Invert space to place label to the left
                    space *= -1
                    # Horizontally align label at right
                    ha = 'right'

                # Use X value as label and format number with one decimal place
                label = "{:.1f}".format(x_value)

                # Create annotation
                plt.annotate(
                    label,                      # Use `label` as label
                    (x_value, y_value),         # Place label at end of the bar
                    xytext=(space, 0),          # Horizontally shift label by `space`
                    textcoords="offset points", # Interpret `xytext` as offset in points
                    va='center',                # Vertically center label
                    ha=ha)                      # Horizontally align label differently for
                                                # positive and negative values.

            plt.tight_layout()
            fig.subplots_adjust(top=0.93)
            plt.show()