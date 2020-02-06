import pandas as pd

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

def writeToFile(outFile, data):
    with open(outFile, 'w+') as file:
        for row in data:
            file.write(row + '\n')