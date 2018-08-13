## Random Forest

```Python
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 02:35:37 2017

@author: Administrator
"""


#%reset -f
# Load the library with the iris dataset

from sklearn.datasets import load_iris
import os
os.chdir('D:\\UserData\\Personal')
# Load scikit's random forest classifier library
from sklearn.ensemble import RandomForestClassifier

# Load pandas
import pandas as pd

# Load numpy
import numpy as np
from random import randrange
from math import sqrt
import matplotlib.pyplot as plt
from collections import Counter

# Set random seed
np.random.seed(0)
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df.head()
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
df.head()
df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75
train, test = df[df['is_train']==True], df[df['is_train']==False]
print('Number of observations in the training data:', len(train))
print('Number of observations in the test data:',len(test))
features = df.columns[0:4]
features
y = pd.factorize(train['species'])[0]



clf = RandomForestClassifier(n_jobs=2, random_state=0,)
clf.fit(train[features], y)
predict=clf.predict(test[features])
clf.predict_proba(test[features])[0:10]

preds = iris.target_names[clf.predict(test[features])]



human_df=pd.read_csv('train_human_python.csv',header=None)

##split it into n folds
def cross_validation_split(dataset, n_folds):
    dataset_split=list()
    len_obs=len(dataset)
    data_copy=pd.DataFrame.copy( human_df)
    index=np.arange(len(dataset))
    np.random.shuffle(index)
    fold_size=int(len(dataset)/n_folds)
    for i in range(n_folds):
        slc_index=index[i*fold_size:(i+1)*fold_size]
        dataset_split.append(data_copy.iloc[slc_index])
    return(dataset_split)


dataset_split=cross_validation_split(human_df,10)


##tune parameter

mean_accu_list=list()
tree_feature_index=list()
clf_list=list()
for n_trees in range(30,41):
    for n_features in range(15,30):

        n_trees=1000
        n_features=23
        #n_features=int(sqrt(len(human_df.columns)-1))
        Accuracy_list=list()
        ##cross validation for 10-fold
        for i in range(10):
            data_copy=list(dataset_split)
            test_set=data_copy[i]
            del data_copy[i] 
            train_set=pd.concat(data_copy)
            len_col=len(train_set.columns)
            test_label=list(test_set.iloc[:,-1])
            train_label=train_set.iloc[:,-1]
            
            ###get train_set without label
            train_set=train_set.loc[:,0:len_col-2]
            test_set=test_set.loc[:,0:len_col-2]
            
            ###random forest model
            clf = RandomForestClassifier(n_estimators=n_trees,n_jobs=2, random_state=0,max_features=n_features,
                                         oob_score=True,max_depth=30)
            clf.fit(train_set,train_label)
            clf_list.append(clf)
            predict=clf.predict(test_set)
        
            
            ###caculate accuracy
            correct=0
            for j in range(len(predict)):
                if predict[j]==test_label[j]:
                    correct+=1
            accuracy=float(correct/len(predict))*100
            Accuracy_list.append(accuracy)
        mean_accu=float(sum(Accuracy_list)/len(Accuracy_list))
        mean_accu_list.append(mean_accu)
        tree_feature_index.append(float(n_trees*1000+n_features)  )
        print('Number of trees: ', n_trees)
        print('Features: ', n_features)
        print('Scores: ' ,Accuracy_list)
        print('Mean Accuracy: ', mean_accu)
        
        
    
plt.plot(tree_feature_index,mean_accu_list)
plt.show()  
max(mean_accu_list)    
a=mean_accu_list
for i in range(len(a)):
    if a[i] >= max(a):
        maxIndex = i
        
        
mean_accu_list[118]
tree_feature_index[118]

###################3
####
######   n_trees=37
####     n_features=28
##
#####################33



####this is used to do leave-one people(300 observations) 
#validation test
human_df=pd.read_csv('train_human_python.csv',header=None)
test_set=pd.read_csv('data_test_updated.csv',header=None)
train_subject=pd.read_csv('training_subjects.csv',header=None)

train_set=human_df.loc[:,0:560]
train_label_new=human_df.iloc[:,-1]

randrange(18)+1

##function to split dataset
def split_by_cate(dataset, a):
    #a = randrange(18)+1
    index_list=list()
    for i in range(len(dataset)):
        index_list.append( (train_subject.iloc[i,-1]==a))
    test_set=dataset[index_list]
    train_set=dataset[[not i for i in index_list]]
    return [train_set, test_set]
    

for a in range(1,19):
    train_set, test_set=split_by_cate(human_df, 3)
    train_label=train_set.iloc[:,561]
    train_set=train_set.iloc[:,0:560]
    test_set_label=list(test_set.iloc[:,561])
    test_set=test_set.iloc[:,0:560]

    clf = RandomForestClassifier(n_estimators=1000,n_jobs=2, 
                                    random_state=0, max_features=28, max_depth=30)
    clf.fit(train_set,train_label)
    predict=clf.predict(test_set)
    predict
    correct=0
    for j in range(len(predict)):
        if (predict[j]==test_set_label[j]):
            correct+=1
    accuracy=float(correct/len(predict))*100
    print('Accuracy',accuracy)

####found they have the lowest accurracy
#1 83
#3 87
#7 78
#9 81
#12 81



#######10/04 Autumn Festival
##To write prediction based on the training set of similar people
###but this failed
human_df=pd.read_csv('train_human_python.csv',header=None)
test_set=pd.read_csv('data_test_updated.csv',header=None)
train_subject=pd.read_csv('training_subjects.csv',header=None)

train_set=human_df.loc[:,0:560]
train_label_new=human_df.iloc[:,-1]

######Delete subject 1

a=1
index_list=list()
for i in range(len(train_subject)):
    index_list.append( (train_subject.iloc[i,-1]==a))
test_set_01=train_set[[i for i in index_list]]
test_set_01_ID=train_subject[[i for i in index_list]]
test_set_01_label=train_label_new[[i for i in index_list]]
train_set_rest=train_set[[not i for i in index_list]]
train_set_rest_ID=train_subject[[not i for i in index_list]]
train_set_rest_label=train_label_new[[not i for i in index_list]]


clf = RandomForestClassifier(n_estimators=37,n_jobs=2, 
                                random_state=0, max_features=28)
train_set_rest_ID=list(train_set_rest_ID.iloc[:,0])
clf.fit(train_set_rest,train_set_rest_ID)
##predict for subject 1, which people of the rest have same activity like him
predict=clf.predict(test_set_01)
count=Counter(predict)
count.most_common()
##
#6 13 17 16 15 9 5 3
similar_most=5
for similar_most in range(1,16):
    similar_list=list()
    #select the similar ID
    for i in range(similar_most):
        similar_list.append(count.most_common()[i][0])
    ##choose the train_set again with these simliar people  
    
    ##bb is temporary storage of index list
    bb=train_subject.isin(similar_list)
    filter_train_set=train_set[list(bb.iloc[:,0])]
    filter_train_set_label=train_label_new[list(bb.iloc[:,0])]
    filter_clf= RandomForestClassifier(n_estimators=200,n_jobs=2,\
                                       random_state=0, max_features=23)
    filter_clf.fit(filter_train_set, filter_train_set_label)
    
    predict=filter_clf.predict(test_set_01)
    predict
    correct=0
    for j in range(len(predict)):
        if (predict[j]==list(test_set_01_label)[j]):
            correct+=1
    accuracy=float(correct/len(predict))*100
    print('Accuracy',accuracy)

################################################
####Above test failed!!!!!!!!!!!
###select similar most 4 people




#######################################################
#################################
######################This time choose the similar people based on activity 1,2, 3....
######################
human_df=pd.read_csv('train_human_python.csv',header=None)
test_set=pd.read_csv('data_test_updated.csv',header=None)
train_subject=pd.read_csv('training_subjects.csv',header=None)

train_set=human_df.loc[:,0:560]
train_label_new=human_df.iloc[:,-1]

######Delete subject 1

a=1
index_list=list()
for i in range(len(train_subject)):
    index_list.append( (train_subject.iloc[i,-1]==a))
test_set_01=train_set[[i for i in index_list]]
test_set_01_ID=train_subject[[i for i in index_list]]
test_set_01_label=train_label_new[[i for i in index_list]]
train_set_rest=train_set[[not i for i in index_list]]
train_set_rest_ID=train_subject[[not i for i in index_list]]
train_set_rest_label=train_label_new[[not i for i in index_list]]



for activity_num in [1:6]:
    #based on acitivity 1 we choose the similar people

##########################################
##########################################
##########################################
##########################################
##########################################
##########################################

#test set ID1 is most similar to ID6
train_set=human_df.loc[:,0:560]
train_set_comb_ID=pd.concat([train_set,train_subject],axis=1, ignore_index=True )
a=1
index_list=list()
for i in range(len(train_subject)):
    index_list.append( (train_subject.iloc[i,-1]==a))
test_01_set=train_set_comb_ID[[i for i in index_list]]
test_set_01_label=train_label_new[[i for i in index_list]]
train_set_rest=train_set_comb_ID[[not i for i in index_list]]
train_set_rest_label=train_label_new[[not i for i in index_list]]

###mostcommon people is 6
for i in range(len(test_01_set)):
    test_01_set.set_value(i,561,6)
    
clf = RandomForestClassifier(n_estimators=1000,n_jobs=2, 
                                random_state=0, max_features=28)

clf.fit(train_set_rest,train_set_rest_label)   
predict=clf.predict(test_01_set)
predict
correct=0
for j in range(len(predict)):
    if (predict[j]==list(test_set_01_label)[j]):
        correct+=1
accuracy=float(correct/len(predict))*100
print('Accuracy',accuracy)





###########################################
####this time we do one-hot encoding
############################
human_df=pd.read_csv('train_human_python.csv',header=None)
test_set=pd.read_csv('data_test_updated.csv',header=None)
train_subject=pd.read_csv('training_subjects.csv',header=None)

train_set=human_df.loc[:,0:560]
train_label_new=human_df.iloc[:,-1]

train_set=human_df.loc[:,0:560]

a=1
index_list=list()
for i in range(len(train_subject)):
    index_list.append( (train_subject.iloc[i,-1]==a))
test_set_01=train_set[[i for i in index_list]]
test_set_01_ID=train_subject[[i for i in index_list]]
test_set_01_label=train_label_new[[i for i in index_list]]
train_set_rest=train_set[[not i for i in index_list]]
train_set_rest_ID=train_subject[[not i for i in index_list]]
train_set_rest_label=train_label_new[[not i for i in index_list]]



hot_col=set(train_set_rest_ID.iloc[:,0])
onehot_encoding=pd.DataFrame(0, index=np.arange(len(train_set_rest_ID)), 
                             columns=np.arange(len(hot_col)))
onehot_encoding.columns=list(hot_col)
for i in range(len(train_set_rest_ID)):
    a=train_set_rest_ID.iloc[i,0]
    onehot_encoding.set_value(i,a,1)
    
###Reset rownames
onehot_encoding = onehot_encoding.reset_index(drop=True)
train_set_rest= train_set_rest.reset_index(drop=True)
train_set_rest_label=train_set_rest_label.reset_index(drop=True)
##We need to reset the column names of train_set_rest and onehot_encoding    
train_set_rest_comb=pd.concat([train_set_rest, onehot_encoding], axis=1)
#reset column names
train_set_rest_comb.columns=range(len(train_set_rest_comb.columns))
train_set_rest_comb



#Now we do onehot encoding for the test set. code those simliar people as 1.
hot_col=set(train_set_rest_ID.iloc[:,0])
onehot_encoding=pd.DataFrame(0, index=np.arange(len(test_set_01_ID)),
                              columns=np.arange(len(hot_col)))
onehot_encoding.columns=list(hot_col)
similar_most=10
similar_list=list()
#select the similar ID
for i in range(similar_most):
    similar_list.append(count.most_common()[i][0])
for i in range(len(test_set_01_ID)):
    for j in similar_list:
        onehot_encoding.set_value(i,j,1)
#reset rownames
onehot_encoding = onehot_encoding.reset_index(drop=True)
test_set_01= test_set_01.reset_index(drop=True)
test_set_01_label=test_set_01_label.reset_index(drop=True)
test_set_01_comb=pd.concat([test_set_01, onehot_encoding], axis=1)
#reset columnnames
test_set_01_comb.columns=range(len(test_set_01_comb.columns))



####from above tidy up, we have 
#taining set: train_set_rest_comb
#training set response: train_set_rest_label
#test set:test_set_01_comb
#test set labels: test_set_01_label

#fit the model
clf = RandomForestClassifier(n_estimators=100,n_jobs=2, 
                                random_state=0, max_features=28)

clf.fit(train_set_rest_comb,train_set_rest_label)   
predict=clf.predict(test_set_01_comb)
predict
correct=0
for j in range(len(predict)):
    if (predict[j]==list(test_set_01_label)[j]):
        correct+=1
accuracy=float(correct/len(predict))*100
print('Accuracy',accuracy)

############################fail again!!







###################
###################
##write it to csv file
predict=clf_list[14].predict(test_set)

dictframe={"Id":np.arange(1,len(predict)+1),
           "Prediction":predict
        }
result_frame=pd.DataFrame(dictframe)

result_frame.to_csv("RF_test_1011.csv",index=False)



```
