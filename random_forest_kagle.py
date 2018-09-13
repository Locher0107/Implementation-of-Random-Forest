# -*- coding: utf-8 -*-
"""
Spyder Editor

Random forest model from scratch
"""




##F9 excute current line
import csv
import random
import math
import operator





##########################################
######Random Forest Algorithm
###################
from random import seed
from random import randrange
from csv import reader
from math import sqrt
import os
#change the current directory
os.chdir('D:\\UserData\\Personal')
#get the current directory
#os.getcwd()

#load a CSV file
def load_csv(filename):
    dataset=list()
    with open(filename,'r') as file:
        csv_reader=reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset
# Convert string column to float
def str_column_to_float(dataset,column):
    #"row" selected each sub-list in a list
    for row in dataset:
        row[column]=float(row[column].strip())
        
#Convert string column to integer
def str_column_to_int(dataset,column):
    class_values=[row[column] for row in dataset]
    unique=set(class_values)
    lookup=dict()
    #we only have two values, M and R
    #set up a look up directory
    for i, value in enumerate(unique):
        #one is set to 0
        #one is set to 1
        lookup[value]=i
    for row in dataset:
        row[column]=lookup[row[column]]
    return lookup

#split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    dataset_split=list()
    #list(dataset) or =dataset they are both same
    dataset_copy=list(dataset)
    fold_size=int(len(dataset)/n_folds)
    for i in range(n_folds):
        fold=list()
        while len(fold)<fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    #the size of data_split is n_folds,
    #each element is a data set
    return dataset_split

#calculate accuracy percentage
def accurracy_metric(actual,predicted):
    correct=0
    for i in range(len(actual)):
        if actual[i]==predicted[i]:
            correct+=1
    return correct/float(len(actual))*100

#evaluating an algorithm using cross validation split
#we can convert function to a parameter of a function
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    #what is *args?
    folds=cross_validation_split(dataset,n_folds)
    scores=list()
    #choose each data set in folds list
    #if we split into 10 dataset 1 fold is 1/10 of total data set
    for fold in folds:
        train_set=list(folds)
        train_set.remove(fold)
        #remove list from list of lists
        train_set=sum(train_set,[])
        test_set=list()
        for row in fold:
            row_copy=list(row)
            test_set.append(row_copy)
           #-1 means the last line
           #why do we set it to be none?
            row_copy[-1]=None
        predicted=algorithm(train_set,test_set, *args)
        actual=[row[-1] for row in fold]
        accurracy=accurracy_metric(actual,predicted)
        scores.append(accurracy)
    return scores
            
#split a dataset based on an attribute and an attribute value
def test_split(index, value,dataset):
    left, right=list(),list()
    for row in dataset:
        if row[index]<value:
            left.append(row)
        else:
            right.append(row)
    return left, right
    #how to choose left and right?

#Calculated the Gini Index for a split dataset
#what is groups?
def gini_index(groups,classes):
    #count all samples at split point
    n_instances=float(sum([len(group) for group in groups ]))
    #sum weighted Gini index for each group
    gini=0.0
    #how many group?
    for group in groups:
        size=float(len(group))
        #avoid divide by zero
        if size==0:
            continue
        score=0.0
        #score the  group based on the score for each class
        for class_val in classes:
            p=[row[-1] for row in group].count(class_val)/size
            score+= p*p
        #weight the group score by its relative size
        gini +=(1.0-score)*(size/n_instances)
    return gini


#get the best split point for a dataset

def get_split(dataset, n_features):
    #why we set it to be list
    class_values=list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups=999,999,999,None
    features=list()
    while len(features)<n_features:
        index=randrange(len(dataset[0])-1)
        if index not in features:
            features.append(index)
    #index selected from the whole range of features
    for index in features:
        for row in dataset:
            #groups=left, right
            groups=test_split(index, row[index],dataset)
            #how to use gini_index to select the split point?
            gini=gini_index(groups,class_values)
            if gini<b_score:
                b_index,b_value,b_score,b_groups=index,row[index],gini,groups
    #return a dict
    return {'index':b_index, 'value':b_value, 'groups':b_groups}

##how to predict?
#count?
def to_terminal(group):
    outcomes=[row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)

#create child splits for a node or make terminal
#node is a dict
def split(node, max_depth, min_size, n_features, depth):
    #node is a dict with 'index' 'value' and 'groups'
    #node['groups'] has two list
    #assign left, right from groups list 1 and list 2
    left, right=node['groups']
    del(node['groups'])
    #check for a no split
    if not right or not left:
        #why?
        node['left']=node['right']=to_terminal(left+right)
        #list 'left' + list 'right' means that combine the list
        return
    #check for max depth
    if depth>=max_depth:
        node['left'],node['right']=to_terminal(left), to_terminal(right)
        return
    #process left child
    if len(left)<=min_size:
        node['left']=to_terminal(left)
    else:
        #what does get_split do?
        node['left']=get_split(left,n_features)
        split(node['left'], max_depth, min_size, n_features, depth+1)
    #process ritht child
    if len(right)<=min_size:
        node['right']=to_terminal(right)
    else:
        node['right']=get_split(right,n_features)
        split(node['right'], max_depth, min_size, n_features, depth+1)
        
#buid a decision tree
def build_tree(train, max_depth,min_size, n_features):
    root=get_split(train, n_features)
    split(root, max_depth, min_size, n_features, 1)
    return root
      
#make a prediction with a decision tree
def predict(node, row):
    if row[node['index']]<node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'],row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'],row)
        else:
            return node['right']
        
#create a random sample from the dataset with replacement
#why do we need a random sample?
def subsample(dataset, ratio):
    sample=list()
    n_sample=round(len(dataset)*ratio)
    while(len(sample)<n_sample):
        index=randrange(len(dataset))
        sample.append(dataset[index])
    return sample

#make a predicition with a list of bagged trees
def bagging_predict(trees, row):
    predictions=[predict(tree, row) for tree in trees]
    return max(set(predictions), key=predictions.count  )

##Random forest algorithm
def random_forest(train, test, max_depth, min_size, sample_size, 
                  n_trees, n_features):
    trees=list()
    for i in range(n_trees):
        sample=subsample(train, sample_size)
        tree=build_tree(sample, max_depth, min_size, n_features)
        trees.append(tree)
    predictions=[bagging_predict(trees, row) for row in test]
    return(predictions)


#Test the random forest algorithm
seed(2003)
#load dataset
filename = 'sonar.all-data.csv'
dataset = load_csv(filename)
#Convert string attributes to floats
for i in range(0, len(dataset[0])-1):
    str_column_to_float(dataset, i)
#Convert class column to integers
str_column_to_int(dataset, len(dataset[0])-1)
#evaluate algorithm
n_folds=5
max_depth=10
min_size=1
sample_size=1.0
n_features=int(sqrt(len(dataset[0])-1 ))
for n_trees in [1,5,10,15,20]:
    scores=evaluate_algorithm(dataset, random_forest, n_folds, max_depth, 
                              min_size, sample_size,n_trees, n_features)
    print('Trees: %d' % n_trees)
    print('Scores: %s'  % scores)
    print('Mean accuracy: %3f%%' %(sum(scores )/float(len(scores))))










len(dataset)
group_0=dataset
node['right']=to_terminal(group_0)

dataset[0]


root=get_split(dataset,4)
left, right=root['groups']
del(root['groups'])
root['left']=root['right']=to_terminal(list())


root['left']=get_split(left, 4)



root
left

groups_0=test_split(2, 0.05,dataset)
groups_0
type(groups_0)

filename = 'sonar.all-data.csv'
dataset = load_csv(filename)
str_columb_to_float(dataset)



ints = [8, 23, 45, 12, 78]
for idx in ints:
    print(idx)
