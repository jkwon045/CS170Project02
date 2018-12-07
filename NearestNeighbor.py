import numpy as np
import math
import sys
import copy
import random

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
    f.close()

    return classification, features

def printFeatures(list_of_features):
    for i in list_of_features:
        print(i+1, end = ' ')
    print('')

#defining leave one out cross validation
def leave_one_out_cross_validation(data, features):
    return random.random()*100

#defining forward search
def forward_search(data):
    classification = data[0]
    features = data[1]

    current_set_of_features = []

    final_accuracy = 0  #accuracy of best set overall in all sets
    final_set_of_features = [] #set of features that give best accuracy

    for i in range( 0, len(features[0]) ):
        print( "On the ", i+1, "th level of the search tree" )
        feature_to_add_at_this_level = -1 #initialize to invalid value
        best_so_far_accuracy = 0
        for j in range ( 0 , len(features[0])):
            if ( j not in current_set_of_features ):
                print("--Considering adding the ", j+1, "th feature")
                temp_features = copy.deepcopy(current_set_of_features)
                temp_features.append(j)
                accuracy = leave_one_out_cross_validation(data, temp_features)
                print( "Accuracy: ", accuracy )
            if ( best_so_far_accuracy < accuracy ):
                best_so_far_accuracy = accuracy
                feature_to_add_at_this_level = j

    
        current_set_of_features.append(feature_to_add_at_this_level)
        if( final_accuracy < best_so_far_accuracy ):
            final_accuracy = best_so_far_accuracy
            final_set_of_features = copy.deepcopy(current_set_of_features)


        #######
        printFeatures(current_set_of_features)
        #######
        print( "On level ", i+1, " added feature ", feature_to_add_at_this_level+1, " to current set\n" )

    return final_set_of_features, final_accuracy


def main():
    data = readData("CS170_SMALLtestdata__45.txt")
    val = forward_search(data)

    printFeatures(val[0])
    print(val[1])
main()
