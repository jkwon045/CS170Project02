import numpy as np
import sys

def readData(filename):
    classification = list()
    features = list()
    f = open(filename, 'r')
    for line in f:
        removedNL = line.strip("\n")
        vals = removedNL.split(' ')
        removedSpaces = list()
        for i in vals:
            while ( ' ' in i ):
                i.strip()
            if ( i is not '' ):
                removedSpaces.append(i)
        classification.append(float(removedSpaces[0]))
        currentfeatures = list()
        for i in range (1, len(removedSpaces)):
            currentfeatures.append(float(removedSpaces[i]))
        features.append(currentfeatures)
    print(classification)
    print(features)
    f.close()

def main():
    f = readData("CS170_SMALLtestdata__45.txt")
main()
