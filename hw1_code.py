#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import export_graphviz
from io import StringIO
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus

def load_data():
    # Get streams of data from the two files
    data_fake = open("clean_fake.txt", "r")
    data_clean = open("clean_real.txt", "r")
    
    data_array = []
    label_array = []
    vectorizer = CountVectorizer()
    
    # For each line in both files, add it to the data list with its label in the corresponding position in the 
    # label list
    for line in data_fake:
        data_array.append(line)
        label_array.append('fake')
    for line in data_clean:
        data_array.append(line)
        label_array.append('real')
        
    # Break the sentences into individual words and record the presence of each word in each sentence
    vectorized_data = vectorizer.fit_transform(data_array).toarray()
    
    data_fake.close()
    data_clean.close()
    
    # First split the data into 70% train and 30% validation and test
    train_X, test_X, train_Y, test_Y = train_test_split(vectorized_data, label_array, test_size = 0.3)
    # In the 30% of data reserved for validation and test, give half of it for each of validation and test
    test_X, validation_X, test_Y, validation_Y = train_test_split(test_X, test_Y, test_size = 0.5)
    # Return train, validation, test, and unique words
    return train_X, test_X, validation_X, train_Y, test_Y, validation_Y, vectorizer.get_feature_names()


# In[2]:


def select_tree_model():
    train_X, test_X, validation_X, train_Y, test_Y, validation_Y, _ = load_data()
    
    test_depths = [1, 3, 7, 15, 30, 60, 100, 300, 1000]
    
    for i in test_depths:
        # Create a Gini and IG classifier of specified depth
        gini_classifier = DecisionTreeClassifier(criterion = 'gini', max_depth = i, random_state = 0)
        ig_classifier = DecisionTreeClassifier(criterion = 'entropy', max_depth = i, random_state = 0)
        # Train the classifieres
        gini_classifier.fit(train_X, train_Y)
        ig_classifier.fit(train_X, train_Y)
        # Test the classifiers using the validation set
        predicted_gini = gini_classifier.predict(validation_X)
        predicted_ig = ig_classifier.predict(validation_X)
        # For each classifier, compare the predicted sets to the known labels
        print("Accuracy for Gini classifier of depth", i, ":", round(accuracy_score(predicted_gini, validation_Y), 4))
        print("Accuracy for IG classifier of depth", i, ":", round(accuracy_score(predicted_ig, validation_Y), 4))
        print("\n")
        
select_tree_model()


# In[3]:


def use_optimal_classifier():
    train_X, test_X, validation_X, train_Y, test_Y, validation_Y, _ = load_data()
    
    # The optimal classifier was determined to have max depth 100
    ig_classifier = DecisionTreeClassifier(criterion = 'gini', max_depth = 100, random_state = 0)
    # Train, test, and find the accuracy of the classifier
    ig_classifier.fit(train_X, train_Y)
    predicted_ig = ig_classifier.predict(test_X)
    print("Accuracy of optimal classifier on test data:", round(accuracy_score(predicted_ig, test_Y), 4))
    return ig_classifier

classifier = use_optimal_classifier()


# In[4]:


import graphviz 

def display_classifier(classifier):
    # features is the list of words used to split
    _, _, _, _, _, _, features = load_data()
    dotfile = StringIO()
    # Write the visualization to an external file
    dotfile = export_graphviz(classifier, out_file = None,  
                filled=True, rounded=True,
                special_characters=True, feature_names = features, class_names=['fake', 'real'])
    graph = graphviz.Source(dotfile) 
    graph.render("test")


# In[5]:


display_classifier(classifier)


# In[6]:


import numpy as np
import math

def compute_information_gain(classifier, keyword):
    _, _, _, _, _, _, features = load_data()
    
    # Get the set of attribute lists for the nodes
    tree = classifier.tree_
    # Contains the word used to split for each node
    features_struct = tree.feature
    
    # Find the first instance of keyword in the list of headline words
    keyword_id = features.index(keyword)
    keyword_id = np.where(features_struct == keyword_id)[0][0]
    
    # Gives the number of real and fake samples contained under a node
    tree_values = tree.value
    # Gives the left and right node position for a node
    true_struct = tree.children_left
    false_struct = tree.children_right
    # Maps each keyword position in the list of headline words to a node position
    state_struct = tree.__getstate__()['nodes']

    # Get left and right children of node corresponding to keyword
    t_node = true_struct[keyword_id]
    f_node = false_struct[keyword_id]

    # Proportion of real responses in the samples under the keyword node
    split_prop = tree_values[keyword_id][0][0]/(tree_values[keyword_id][0][0] + tree_values[keyword_id][0][1])
    # Find entropy of the keyword node
    prior_entropy = -split_prop * math.log(split_prop, 2) - (1 - split_prop) * math.log(1 - split_prop, 2)

    # Proportion of real responses in the samples under the left and right child of keyword node
    split_t_prop = tree_values[t_node][0][0]/(tree_values[t_node][0][0] + tree_values[t_node][0][1])
    split_f_prop = tree_values[f_node][0][0]/(tree_values[f_node][0][0] + tree_values[f_node][0][1])
    
    # Find entropy of left and right child of keyword node, and if all responses are real or fake, the entropy is 0
    if split_t_prop == 0 or split_t_prop == 1:
        entropy_t_split = 0
    else:
        entropy_t_split = -split_t_prop * math.log(split_t_prop, 2) - (1 - split_t_prop) * math.log(1 - split_t_prop, 2)
        
    if split_f_prop == 0 or split_f_prop == 1:
        entropy_f_split = 0
    else:
        entropy_f_split = -split_f_prop * math.log(split_f_prop, 2) - (1 - split_f_prop) * math.log(1 - split_f_prop, 2)  
    
    # The proportion of samples belonging to the left node
    t_weight = state_struct[t_node][5]/state_struct[keyword_id][5]

    # Weighted average of entropies of left and right children of keyword node
    posterior_entropy = t_weight * entropy_t_split + (1 - t_weight) * entropy_f_split
    information_gain = prior_entropy - posterior_entropy

    print("Information gain on keyword", keyword, ":", information_gain)

keyword_test = ["the", "hillary", "donald", "trump"]

for keyword in keyword_test:
    compute_information_gain(classifier, keyword)


# In[7]:


from sklearn.neighbors import KNeighborsClassifier

def select_knn_model():
    train_X, test_X, validation_X, train_Y, test_Y, validation_Y, _ = load_data()
    
    test_depths = [i for i in range(1, 21)]
    # Train and validation errors for each k value
    train_error_rates = []
    validation_error_rates = []
    
    for i in test_depths:
        # Build and train a KNN classifier with i neighbours
        knn_classifier = KNeighborsClassifier(n_neighbors = i)
        knn_classifier.fit(train_X, train_Y)
        # Results of classifier on train and validation sets
        predicted_train = knn_classifier.predict(train_X)
        predicted_validation = knn_classifier.predict(validation_X)
        # Accuracy of classifier on train and validation sets
        train_error_rate = 1 - round(accuracy_score(predicted_train, train_Y), 4)
        validation_error_rate = 1 - round(accuracy_score(predicted_validation, validation_Y), 4)
        train_error_rates.append(train_error_rate)
        validation_error_rates.append(validation_error_rate)
        print("Training error rate for KNN classifier with", i, "neighbors:", train_error_rate)
        print("Validation error rate for KNN classifier with", i, "neighbors:", validation_error_rate)
        print("\n")
    return train_error_rates, validation_error_rates


# In[10]:


import matplotlib.pyplot as plt
import numpy as np

# For each k from 1 to 20, a train error and validation error will be plotted above it
train_error_rates = [0, 0.1129, 0.1347, 0.1461, 0.1894, 0.1938, 0.2157, 0.2244, 0.2402, 0.2502, 0.2292, 0.2467, 0.2453,
                     0.2498, 0.2445, 0.2515, 0.241, 0.2423, 0.2472, 0.2485]
validation_error_rates = [0.3102, 0.349, 0.3265, 0.3245, 0.3265, 0.3224, 0.2980, 0.3265, 0.3408, 0.3367, 0.3571, 0.3388,
                          0.349, 0.3673, 0.3388, 0.3469, 0.3367, 0.351, 0.3408, 0.3531]
k_values = [i for i in range(1, 21)]
# Plot two separate trends for train and validation errors, each against the k values
plt.plot(k_values, train_error_rates, "bo", k_values, validation_error_rates, "mo", linestyle = "--")
plt.legend(["Train error", "Validation error"])
plt.title("KNN classifier error rates")
# Range ensures no points are cut off
plt.xticks(np.arange(1, 21, 1))
plt.yticks(np.arange(0, 0.45, 0.05))
plt.xlabel("k")
plt.ylabel("Error rate")
plt.show()


# In[9]:


def use_optimal_knn():
    train_X, test_X, validation_X, train_Y, test_Y, validation_Y, _ = load_data()
    
    # The optimal KNN used 7 neighbours
    optimal_knn = KNeighborsClassifier(n_neighbors = 7)
    # Train classifier and apply test set to it
    optimal_knn.fit(train_X, train_Y)
    predicted_knn = optimal_knn.predict(test_X)
    print("Accuracy of KNN with 7 neighbours on test data:", round(accuracy_score(predicted_knn, test_Y), 4))
use_optimal_knn()

