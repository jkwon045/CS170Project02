# NearestNeighbor.py
# Jasmine Kwong
# SID: 862053634
# A feature search algorithm that uses nearest neighbor with leave one out cross validatoin
# to calculate accuracy.

import math
import copy

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

def makeStringList(list_of_features):
    retString = ''
    for i in list_of_features:
        retString += ( str(i+1) + ' ')
    return retString

def euclidean_distance(p1, p2, feature_set):
    dist = 0
    for i in feature_set:
        dist += (p1[i]- p2[i])**2
    return math.sqrt(dist)

#defining leave one out cross validation
def leave_one_out_cross_validation(classification, data, test_feature_set):

    listOfClassifications = []
    numcorrect = 0
    for i in range(len(data)):
        closest_dist = 100
        assumed_class = -1
        for j in range(len(data)):
            if ( i == j ):
                continue
            #calculate the distance from I to J
            temp_dist = euclidean_distance(data[i], data[j], test_feature_set)
            #if this distance is closer than the current closest distance, this is the "class" of i
            if( temp_dist <= closest_dist ):
                closest_dist = temp_dist
                assumed_class = classification[j]
        if( classification[i] == assumed_class ):
            numcorrect+=1
        #after running it over all the data points count if its right or wrong
        #for accuracy 
    return numcorrect/len(data)

#defining forward search
def forward_search(data):
    classification = data[0]
    features = data[1]

    current_set_of_features = []

    final_accuracy = 0  #accuracy of best set overall in all sets
    final_set_of_features = [] #set of features that give best accuracy
    decrease = False
    for i in range( 0, len(features[0]) ):
        print( "On the ", i+1, "th level of the search tree" )
        feature_to_add_at_this_level = -1 #initialize to invalid value
        best_so_far_accuracy = 0
        for j in range ( 0 , len(features[0])):
            if ( j not in current_set_of_features ):
                print("--Considering adding feature ", j+1)
                temp_features = copy.deepcopy(current_set_of_features)
                temp_features.append(j)
                accuracy = leave_one_out_cross_validation(classification, features, temp_features)
                print("Using features ", makeStringList(temp_features), " accuracy was ", accuracy* 100 , "%")
                if ( best_so_far_accuracy < accuracy ):
                    best_so_far_accuracy = accuracy
                    feature_to_add_at_this_level = j

    
        current_set_of_features.append(feature_to_add_at_this_level)
        if ( final_accuracy < best_so_far_accuracy and len(current_set_of_features) > 2):
            final_accuracy = best_so_far_accuracy
            final_set_of_features = copy.deepcopy(current_set_of_features)
        elif ( final_accuracy > best_so_far_accuracy and not decrease):
            decrease = True
            print("(Warning, Accuracy has decreased! Continuing search in case of local maxima)") 

        print( "On level ", i+1, " added feature ", feature_to_add_at_this_level+1, " to current set\n" )

    return final_set_of_features, final_accuracy

def backward_search(data):
    classification = data[0]
    features = data[1]

    current_set_of_features = []

    #Create a set with all the features
    for i in range(0, len(features[0])):
        current_set_of_features.append(i)

    final_accuracy = 0
    final_set_of_features = []
    increase = False
    for i in reversed(range(len(features[0]))):

        print( "On the ", i+1, "th level of the search tree" )
        starting_accuracy = leave_one_out_cross_validation(classification, features, current_set_of_features)
        least_difference = 100
        increase = False
        for j in current_set_of_features:
            print( "Considering removing feature ", j+1 )
            temp_features = copy.deepcopy(current_set_of_features)
            temp_features.remove(j)

            accuracy = leave_one_out_cross_validation(classification, features, temp_features)
            print("Using features ", makeStringList(temp_features), " accuracy was ", accuracy* 100 , "%")
            difference = starting_accuracy - accuracy

            if ( difference < least_difference and difference > 0):
                least_difference = difference
                feature_to_remove = j
            elif ( difference < least_difference ):
                least_difference = difference
                feature_to_remove = j

        current_set_of_features.remove(feature_to_remove)

        if( final_accuracy < ( starting_accuracy - least_difference ) and len(current_set_of_features) > 2):
            final_accuracy = starting_accuracy - least_difference
            final_set_of_features = copy.deepcopy(current_set_of_features)
        
        print( "On level ", i+1, " removed feature ", feature_to_remove+1, " from current set\n" )

    return final_set_of_features, final_accuracy

def leave_one_out_cross_validationImproved(classification, data, test_feature_set, best_acc_so_far):

    listOfClassifications = []
    numcorrect = 0
    numIncorrect = 0
    for i in range(len(data)):
        closest_dist = 100
        assumed_class = -1
        for j in range(len(data)):
            if ( i == j ):
                continue
            #calculate the distance from I to J
            temp_dist = euclidean_distance(data[i], data[j], test_feature_set)
            #if this distance is closer than the current closest distance, this is the "class" of i
            if( temp_dist <= closest_dist ):
                closest_dist = temp_dist
                assumed_class = classification[j]
        if( classification[i] == assumed_class ):
            numcorrect+=1
        else:
            numIncorrect+=1

        if(numIncorrect > (len(data) - (best_acc_so_far*len(data)))):
            return 0
        #after running it over all the data points count if its right or wrong
        #for accuracy 
    return numcorrect/len(data)

def improved_search(data):
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
                print("--Considering adding feature ", j+1)
                temp_features = copy.deepcopy(current_set_of_features)
                temp_features.append(j)
                accuracy = leave_one_out_cross_validationImproved(classification, features, temp_features, best_so_far_accuracy)
                print("Using features ", makeStringList(temp_features), " accuracy was ", accuracy* 100 , "%")
                if ( best_so_far_accuracy < accuracy ):
                    best_so_far_accuracy = accuracy
                    feature_to_add_at_this_level = j
                else:
                    continue
    
        current_set_of_features.append(feature_to_add_at_this_level)
        if( final_accuracy < best_so_far_accuracy and len(current_set_of_features) > 2):
            final_accuracy = best_so_far_accuracy
            final_set_of_features = copy.deepcopy(current_set_of_features)

        print( "On level ", i+1, " added feature ", feature_to_add_at_this_level+1, " to current set\n" )

    return final_set_of_features, final_accuracy

def main():
    print("Welcome to Bertie Woosters Feature Selection Algorithm.")
    f = input("Type in the name of the file to test: ")
    data = readData(f)
    
    print("Please type the number of the algorithm you want to run")
    print("\t 1) Forward Selection" )
    print("\t 2) Backward Elimination" )
    print("\t 3) Bertie's Special Algorithm")

    alg = int(input())

    print("This data set has ", len(data[1][0]), " features (not including the class attribute), with ",\
    len(data[0]), " instances.")

    val = list()

    if( alg == 1 ):
        val = forward_search(data)
    elif ( alg == 2 ):
        val = backward_search(data)
    else:
        val = improved_search(data)
        
    print("Finished search!")
    print("The best feature subset is ", makeStringList(val[0]), "which has an accuracy of ", val[1]*100, "%.")

main()
