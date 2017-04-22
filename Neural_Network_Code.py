# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 11:44:24 2017

@author: Abdullah Mobeen
"""
import math
import time
        
def neuron(prev_act,input_weights):
    """Single Neuron Unit. Takes as input previous activatios and input weights.
    Returns the new activation value after applying the sigmoid function"""
    
    z = 0
    for i in range(len(prev_act)):
        dot = prev_act[i]*input_weights[i]
        z += dot
    return (1/(1+math.exp(-z)))

def files(data):
    """Takes as input the file name and processes the file to return 
    vectors and length of vectors"""
    
    with open(data,'r') as f:
        w = []
        a = f.readlines()
        for i in a:
            i = i.split(',')
            w.append(i)
        for i in range(len(w)):
            for j in range(len(w[0])):
                w[i][j] = float(w[i][j])
        return w, len(w)

def data_processing():
    """Calls files(data) function to process all the data files provided in the 
    Homework. Returns weights between layers, hidden units, output units,
    image data, and labels"""
    
    bias = [1.0]   
    x_vector = files('ps5_data.csv')[0] # x_vector = list of pixels for each image
    
    for i in range(5000):
        x_vector[i] = bias + x_vector[i]
        
    with open('ps5_data-labels.csv','r') as f:
        labels = f.readlines()  #labels = list of labels with which predicted labels will be compared
    for i in range(len(labels)):
        labels[i] = int(labels[i])-1
        
    weights, total_nodes = files('ps5_theta1.csv')
    # weights = array of 25 lists, each list has 401 weights
    # total_nodes = total no. of nodes in the hidden layer
    weights1, total_nodes1 = files('ps5_theta2.csv')
    # weights1 = array of 10 lists, each list has 26 weights
    # total_nodes = total no. of nodes in the output layer
    
    return weights, total_nodes, weights1, total_nodes1, x_vector, labels

        
def forward_propogation(x):
    """Forward Propogation function that takes vector for one single image
    as input and returns 10 outputs at output units as one single list"""
    
    global weights, total_nodes, weights1, total_nodes1
    
    l = [1.0]
    final_l = []
    
    for i in range(total_nodes):
        a = neuron(x,weights[i])
        l.append(a)
        
    for i in range(total_nodes1):
        a = neuron(l,weights1[i])
        final_l.append(a)
        
    return final_l            
            
def image_classifier(x):
    """Takes as input vector of one single image and classifies it as a number 
    between 0-9. Returns the prediction""" 
    
    l = forward_propogation(x)
    index = l.index(max(l))
    return index

def classification():
    """Function to classify all 5000 images.
    Returns the error rate on the entire classification"""
    
    global x, labels
    
    classification_list = []
    for i in range(len(x)):
        prediction = image_classifier(x[i])
        classification_list.append(prediction)
    errors = 0 #counter for errors made
    for i in range(len(classification_list)):
        if classification_list[i] != labels[i]:
            errors += 1
    return (errors/len(labels))*100

def label_normalization(labels):
    """Takes as input the labels already provided between 0-9.
    Returns an array that contains a vector for each label.
    For example: if the number is 0, the vector is [1,0,0,0,0,0,0,0,0,0]"""
    
    l = labels
    for i in l:
        val = i #temporarily stores the value i to be later used as index
        x = l[l.index(i)] = [0 for j in range(10)] #changes all the labels to vectors [0,0,0,0,0,0,0,0,0,0]
        x[val] = 1 #updates the specific unit to 1
    return l
    
    
def cost_function(labels, x):
    """Takes as input labels and the image data - x. 
    Applies the MLE cost function to calculate the cost.
    Returns the cost"""
    
    global weights, weights1
    new_labels = label_normalization(labels)
    predicted_labels = [] #list to collect the predicted labels
    for i in range(len(new_labels)):
        l = forward_propogation(x[i])
        predicted_labels.append(l)
    cost = 0
    for i in range(5000):
        for k in range(10):
            c = float(-new_labels[i][k]) * (math.log(predicted_labels[i][k]))\
            - (float(1.0 - new_labels[i][k])) * math.log(1.0 - predicted_labels[i][k])
            cost += c
    cost = cost*(1/len(predicted_labels))
#    reg = 0   #Regularization term as normally used in Neural Network Cost functions
#    for i in weights:
#        for j in i:
#            reg = reg + j**2
#    for i in weights1:
#        for j in i:
#            reg = reg + j**2
#    reg = reg/(2*len(new_labels))
#    cost = cost*(1/len(predicted_labels))  + reg #Cost function with the regularization term
    return cost

if __name__ == "__main__":
    
    weights, total_nodes, weights1, total_nodes1, x, labels = data_processing()
    start_time = time.time()
    print("The error rate when classifying 5000 images: ",classification(),"%\n")
    print("Total time taken for classification: ", round(time.time() - start_time,2), 'seconds\n')
    print("The cost function correct to 4 d.p: ", round((cost_function(labels,x)),4))
