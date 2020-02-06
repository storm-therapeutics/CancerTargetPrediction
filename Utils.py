import pandas as pd
import pandas
import pickle

def obtainOutputFileName(outputPath, cancerType):
    return outputPath + cancerType.capitalize() + "/" + cancerType + "_model_selection_statistics.csv"

def obtainPredictionsOutputFileName(outputPath, cancerType):
    return outputPath + cancerType.capitalize() + "/" + cancerType + "_model_selection_predictions.csv"

def obtainOutputModelFileName(outputPath, cancerType, modelName, sampleId):
    return outputPath + cancerType.capitalize() + "/" + cancerType + "_" + modelName + "_s" + sampleId + ".pkl"

def getDatasets(dataFolder, cancerType, available_samples):
    dataframes = []
    featureSets = getFeatureSet(cancerType, available_samples)

    for inx, sample in enumerate(available_samples):
        dataframe = pandas.read_csv(dataFolder + cancerType.capitalize() + "/" + cancerType + "_training_data_" + sample + ".dat", header=0, sep=",")

        dataframe = dataframe[dataframe['label'] != 2]
        dataframe.drop("gene", axis=1, inplace=True)

        dataframePositive = dataframe[dataframe['label'] == 1]
        dataframeNegative = dataframe[dataframe['label'] == 0]

        positiveSize = dataframePositive.shape[0]
        negativeSize = dataframeNegative.shape[0]

        # Set them the same size
        if(positiveSize > negativeSize):
            dataframePositive = dataframePositive.head(-(positiveSize-negativeSize))
        elif(negativeSize > positiveSize):
            dataframeNegative = dataframeNegative.head(-(negativeSize-positiveSize))

        dataframe = pd.concat([dataframePositive, dataframeNegative])  
        dataframeFeatureSet = list(featureSets[0])
        dataframeFeatureSet.append("label")
        dataframe = dataframe[dataframeFeatureSet]

        dataframes.append(dataframe)
    
    return dataframes

def getFeatureSet(cancerType, available_samples):
    # Read feature importance file
    featureSubsets = []
    dataframe = pandas.read_csv("output/feature_importance/" + cancerType + "_feature_importance.csv", header=0, sep=",")
    dataframe = dataframe.loc[dataframe['z_score'] >= 0.5]
    featureSubsets.append(dataframe['feature'].values)

    return featureSubsets

def select_best_method(modelPerformancesFolder, dataFile, cancerType, available_samples, useAveraged=True):
    modelSelectionAttribute = "auc"

    # Get the data for the first feature selection approach: one feature set for all (average)
    statisticsFilePath = modelPerformancesFolder + "/" + cancerType + "_model_selection_statistics.csv"
    statisticsDataframe = pandas.read_csv(statisticsFilePath, header=0, sep=",")
    statisticsDataframe = statisticsDataframe[statisticsDataframe['sampleid'] == "overall"]
    statisticsDataframe = statisticsDataframe.ix[statisticsDataframe[modelSelectionAttribute].idxmax()]

    # Get the best approach and its model name
    bestModel = (statisticsDataframe["model"], statisticsDataframe[modelSelectionAttribute])

    # Get the path for the models
    modelsPath = modelPerformancesFolder + "/" + cancerType + "_" + bestModel[0] + "_XX.pkl"

    # Save the models in an array
    models_to_return = []

    feature_sets = getFeatureSet(cancerType, available_samples)

    positiveGenesSet = list()
    negativeGenesSet = list()
    genesSetLabels = list()
    
    for sampleInx, sample in enumerate(available_samples):
        sampleModelPath = modelsPath.replace("XX", sample)
        with open(sampleModelPath, 'rb') as f:
            sampleModel = pickle.load(f)

            # Prepare data in case of refit
            dataframe = pandas.read_csv(dataFile + "/" + cancerType + "_training_data_" + sample + ".dat", header=0, sep=",")
            dataframe = dataframe[dataframe['label'] != 2]
            
            dataframePositive = dataframe[dataframe['label'] == 1]
            dataframeNegative = dataframe[dataframe['label'] == 0]

            positiveSize = dataframePositive.shape[0]
            negativeSize = dataframeNegative.shape[0]

            # Set them the same size
            if(positiveSize > negativeSize):
                dataframePositive = dataframePositive.head(-(positiveSize-negativeSize))

            elif(negativeSize > positiveSize):
                dataframeNegative = dataframeNegative.head(-(negativeSize-positiveSize))

            for geneName in dataframePositive['gene'].values:
                if geneName not in positiveGenesSet:
                    positiveGenesSet.append(geneName)       
                    genesSetLabels.append("positive")      
  
            for geneName in dataframeNegative['gene'].values:
                if geneName not in negativeGenesSet:
                    negativeGenesSet.append(geneName) 
                    genesSetLabels.append("negative")

            dataframe = pd.concat([dataframePositive, dataframeNegative])


            dataframe.drop("gene", axis=1, inplace=True)
            dataframeNoLabel = dataframe[feature_sets[0]]
            dataset = dataframe.values
            X = dataframeNoLabel.astype(float)
            Y = dataframe['label'].values

            # Refit with whole data if it's not LR
            if bestModel[0] not in ["lr"]: 
                sampleModel.fit(X, Y)

            models_to_return.append(sampleModel)

    positiveGenesSet = list(set(positiveGenesSet))
    negativeGenesSet = list(set(negativeGenesSet))

    finalGenesSet = positiveGenesSet + negativeGenesSet

    return models_to_return, bestModel[0], finalGenesSet

def getDataForPrediction(dataFile, cancerType, available_samples, genesTrainingSet=None):
    feature_sets = getFeatureSet(cancerType, available_samples)
    data_for_prediction = []

    for fs in feature_sets:
        filePath = dataFile + "/" + cancerType + "_training_data_s1.dat"
        dataframe = pandas.read_csv(filePath, header=0, sep=",")
        dataframe = dataframe.sort_values(by=['label'])

        # Keep only unknown class genes
        dataframe = dataframe[~dataframe['gene'].isin(genesTrainingSet)]
        geneNames = dataframe["gene"].values
        dataframe = dataframe[fs]

        # Get numerical X data
        dataset = dataframe.values
        X = dataset[:,0:dataset.shape[1]].astype(float)

        data_for_prediction.append((geneNames, X))

    return data_for_prediction

def writeToFile(outFile, data):
    with open(outFile, 'w+') as file:
        for row in data:
            file.write(row + '\n')