import sys
import argparse
import json
import requests
from pandas import *
import pickle
import os
import numpy as np
import math
import re
import csv
import warnings
import statistics
import re
import mygene
from opentargets import OpenTargetsClient
ot = OpenTargetsClient()
warnings.filterwarnings("ignore")

cancerData = dict()
cancerData["leukemia"] = "EFO_0000565"
cancerData["lung"] = "MONDO_0008903"
cancerData["ovarian"] = "MONDO_0008170"
cancerData["pancreatic"] = "EFO_0002618"
cancerData["kidney"] = "MONDO_0002367"
cancerData["bladder"] = "MONDO_0001187"
cancerData["breast"] = "MONDO_0007254"
cancerData["colon"] = "EFO_1001950"
cancerData["liver"] = "MONDO_0002691"

def parse_args():
    parser = argparse.ArgumentParser(description = "", epilog = "")
    parser.add_argument("-pf", "--predictionsFile", help="Path to one of the predictions files to get the gene symbols from (REQUIRED).", dest="predictionsFile")
    return parser.parse_args()
	
def writeToFile(filePath, data):
    with open(filePath, 'w+') as file:
        for index, element in enumerate(data):
            file.write(element + '\n')

def getOpenTargetsScores(predictionsFile):
    dataToSave = list()
    dataToSave.append("gene,cancerType,predictionProbability,nrUniquePublications,overallAssociationScore,hasAnyData")
    for cancer in cancerData:
        geneSymbols = list()
        targetProbabilities = dict()

        with open(predictionsFile) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    line_count += 1
                    continue
                else:
                    geneSymbols.append(str(row[0]))
                    targetProbabilities[str(row[0])] = str(round(float(row[1]), 3))
                    line_count += 1
                    if line_count == 11:
                        break
        
        for gene in geneSymbols:
            ensemblIds = list()
            mg = mygene.MyGeneInfo()

            result = mg.query(gene, scopes="symbol", fields=["ensembl"], species="human", verbose=False)
            hgnc_name = gene
            for hit in result["hits"]:
                if "ensembl" in hit and "gene" in hit["ensembl"]:
                    ensemblIds.append(hit["ensembl"]["gene"])

            associationsForTarget = ot.get_associations_for_target(gene)
            if associationsForTarget:
                for assoc in associationsForTarget:
                    geneEnsemblId = assoc['id'].split("-")[0]
                    ensemblIds.append(geneEnsemblId)

            ensemblIds = list(set(ensemblIds))

            publicationsCount = 0
            hasData = "No"
            associationScore = 0.0
			
            for ensId in ensemblIds:
                associationsForTarget = ot.filter_evidence()
                associationsForTarget.filter(target=ensId)
                if associationsForTarget:
                    hasData = "Yes"
                associationsForTarget.filter(disease=cancerData[cancer])
                associationsForTarget.filter(datatype="literature")
				
                publications = list()
                for association in associationsForTarget:
                    publications.append(association["unique_association_fields"]["publication_id"])
                
                publicationsNr = len(list(set(publications)))
                print(publicationsNr)

                if associationsForTarget:
                    # Session expires once you access an element so need to query again 
                    associationsForTarget = ot.filter_associations()
                    associationsForTarget.filter(target=ensId)
                    associationsForTarget.filter(disease=cancerData[cancer])

                    associationScore = round(float(associationsForTarget[0]["association_score"]["overall"]), 3)

            dataToSave.append(gene + "," + cancer + "," + targetProbabilities[gene] + "," + str(publicationsNr) + "," + str(associationScore) + "," + hasData)
        
        # Save to file
        writeToFile("opentargets_evidence_top10targets.csv", dataToSave)

if __name__ == '__main__':
    args = parse_args()     
    getOpenTargetsScores(args.predictionsFile)