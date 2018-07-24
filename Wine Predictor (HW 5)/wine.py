'''
Wendy Wang
UNI: www2105
'''

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt

def euclidean_distance(a,b):
    diff = a - b
    return np.sqrt(np.dot(diff, diff))

def load_data(csv_filename):
    """ 
    Returns a numpy ndarray in which each row repersents
    a wine and each column represents a measurement. There should be 11
    columns (the "quality" column cannot be used for classificaiton).
    """
    ndarray = np.genfromtxt(csv_filename, delimiter=';', skip_header=1)
    ndarray = ndarray[:,:-1]
    
    return ndarray
    
def split_data(dataset, ratio = 0.9):
    """
    Return a (train, test) tuple of numpy ndarrays. 
    The ratio parameter determines how much of the data should be used for 
    training. For example, 0.9 means that the training portion should contain
    90% of the data. You do not have to randomize the rows. Make sure that 
    there is no overlap. 
    """
    num_train = int(round(len(dataset) * ratio))
    train = dataset[:num_train]
    test = dataset[num_train:]
    
    return (train, test)
    
def compute_centroid(data):
    """
    Returns a 1D array (a vector), representing the centroid of the data
    set. 
    """
    return sum(data)/len(data)
    
def experiment(ww_train, rw_train, ww_test, rw_test, output=True):
    """
    Train a model on the training data by creating a centroid for each class.
    Then test the model on the test data. Prints the number of total 
    predictions and correct predictions. Returns the accuracy. 
    """
    ww_centroid = compute_centroid(ww_train)
    rw_centroid = compute_centroid(rw_train)
    
    ww_correct = ww_total = rw_correct = rw_total = 0;
    for row in ww_test:
        if euclidean_distance(ww_centroid, row) < euclidean_distance(rw_centroid, row):
            ww_correct += 1;
        ww_total += 1;
    
    for row in rw_test:
        if euclidean_distance(rw_centroid, row) < euclidean_distance(ww_centroid, row):
            rw_correct += 1;
        rw_total += 1;
    correct = ww_correct + rw_correct
    total = ww_total + rw_total
    accuracy = correct/total
    #w = "Total white wine predictions: {}\nCorrect white win predictions: {}".format(ww_total, ww_correct)
    #r = "Total red wine predictions: {}\nCorrect red win predictions: {}".format(rw_total, rw_correct)
    t = "Total correct: {} Total predictions: {} Accuracy: {}".format(correct,total,accuracy)
    if output:
        print(t)
    return accuracy;
    
    
def learning_curve(ww_training, rw_training, ww_test, rw_test):
    """
    Perform a series of experiments to compute and plot a learning curve.
    """
    np.random.shuffle(ww_training)
    np.random.shuffle(rw_training)
    
    accuracies = []
    for n in range(1, len(ww_training)+1):
        accuracies.append(experiment(ww_training[:n], rw_training[:n], ww_test, rw_test, False))
    num_tr = [i for i in range(1,len(ww_training)+1)]
    plt.xlabel("Number of training items used")
    plt.ylabel("Accuracy")
    plt.plot(num_tr,accuracies)
        
    
    
def cross_validation(ww_data, rw_data, k):
    """
    Perform k-fold crossvalidation on the data and print the accuracy for each
    fold. 
    """
    accuracy_sum = 0;
    ww_split = np.array_split(ww_data,k)
    rw_split = np.array_split(rw_data,k)

    ww_training_arr = np.zeros((1,11))
    rw_training_arr = np.zeros((1,11))
    for index_to_test in range(k): #k==len(ww_split)
        for i in range(k):
            if i != index_to_test:
                ww_training_arr = np.concatenate((ww_training_arr, ww_split[i]))
                rw_training_arr = np.concatenate((rw_training_arr, rw_split[i]))
        #ww_training_arr = ww_training_arr[1:]
        #rw_training_arr = rw_training_arr[1:]
        accuracy_sum += experiment(ww_training_arr, rw_training_arr, ww_split[index_to_test], rw_split[index_to_test], False)
    return accuracy_sum/k
    
    
if __name__ == "__main__":
    
    ww_data = load_data('whitewine.csv')
    rw_data = load_data('redwine.csv')

    # Uncomment the following lines for step 2: 
    ww_train, ww_test = split_data(ww_data, 0.9)
    rw_train, rw_test = split_data(rw_data, 0.9)
    experiment(ww_train, rw_train, ww_test, rw_test)
    
    # Uncomment the following lines for step 3
    ww_train, ww_test = split_data(ww_data, 0.9)
    rw_train, rw_test = split_data(rw_data, 0.9)
    learning_curve(ww_train, rw_train, ww_test, rw_test)
    
    # Uncomment the following lines for step 4:
    k = 10
    acc = cross_validation(ww_data, rw_data,k)
    print("{}-fold cross-validation accuracy: {}".format(k,acc))
    