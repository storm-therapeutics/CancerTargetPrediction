#!/usr/bin/env python3

import numpy as np

cancer_type_list = ["liver","breast","bladder", "colon", "ovarian", "kidney", "leukemia","pancreatic","lung"]
for cancer in cancer_type_list:
    rain = open("output/feature_correlation/" + cancer + "_feature_correlation.csv","r")

    average = 0.0
    total = 0
    maxt = 0.0

    for line in rain:
        try:
            p = np.abs(float(line.split(",")[4]))
            if p == 1.0: continue
            average += p
            total += 1
            maxt = max(maxt,p)
        except:
            pass

    average = average / float(total)

    print("Average:",average)
    print("Maximum:",maxt)
    print("------")
