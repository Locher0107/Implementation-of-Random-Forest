rm(list=ls())



####This script is for the 4 model combination.
#1. xgboost92.6 RF92.2 SVM and Regularized Logistic regression

load("svmgrid_fixedgamma_default.RData")
val_store




rm(list=ls())

##record time
start<-proc.time()
accu<-as.numeric()
#library(foreach)
#library(doParallel)
library(e1071)
#cl_2<-makeCluster(2)
#registerDoParallel(cl_2)


dat<-read.csv("train_human_python.csv", header=FALSE)
train_ID<-read.csv("training_subjects.csv",header=FALSE)
test_data<-read.csv("test_data.csv",header=FALSE)
train_label<-dat[,562]
train_X<-dat[,-562]


#split data function
split_data<-function(train_ID){
  index_1<-sample(1:nrow(train_ID),size=round(0.9*nrow(train_ID)  ) )

  return(index_1)
}


###for loop to tune model
###make a list to store value

val_store<-list()

#C0 form 2^-5 to 2^15
#gamma0 from 2^-15 to 2^3

##
svm_fit<-list()
accu<-as.numeric()
pred<-list()
for(i in 1:10) {
  


    cat(i)

    
    ##split data
    index<-split_data(train_ID)
    X_train<-train_X[index,select_feature]### select_feature is from xgboost
    Y_train<-train_label[index]
    X_test<-test_data[,select_feature]
    
    x_train<-as.matrix(X_train)
    Y_train<-as.factor(Y_train)
    
    svm_fit[[i]]<-svm(x=X_train, y=Y_train, kernel = "radial", cost = 3+rnorm(1,0, 0.001 ) )
    
    pred[[i]]<-predict(svm_fit[[i]], X_test)


}


head(pred[[1]])
head(pred[[3]])



combine_pred<-c()
for(i in 1:3926){
 
  temp_store<-c()
  for(j in 1:10){
    temp_store[j]<-(pred[[j]])[i]

    
    
  }
  combine_pred[i]<-names(sort(table(temp_store),decreasing = TRUE))[1]
  
  
  
  
}


combine_pred<-as.integer(combine_pred)
svm_select_fit<-combine_pred

  



library(dplyr)
library(tibble)
library(tidyverse)
#model_1<-read.csv("model1.csv")#linear svm
View(model_1)
model_2<-read.csv("model2.csv")#xgboost
model_3<-read.csv("model3.csv")#random forest

dim(model_3)
dim(model_2)

vec_1<-as.numeric()
vec_2<-as.numeric()
vec_3<-as.numeric()
count=1
#| model_1[i,2]!=model_3[i,2] 
for(i in 1:length(model_1)){
  if( model_1[i]!=model_2[i,2] | model_1[i]!=model_3[i,2]  ){
    vec_1<-append(vec_1, model_1[i])
    vec_2<-append(vec_2, model_2[i,2])
    vec_3<-append(vec_3, model_3[i,2])
    
    
  }
}

compare_table<-cbind(vec_1,vec_2,vec_3)
head(compare_table)
nrow(compare_table)


###Regularized logistic fit
library(LiblineaR)
dat<-read.csv("train_human_python.csv", header=FALSE)
train_ID<-read.csv("training_subjects.csv",header=FALSE)
test_data<-read.csv("test_data.csv",header=FALSE)
train_label<-dat[,562]
train_X<-dat[,-562]

class(train_label)
train_Y<-as.factor(train_label)
####feature selection
train_X<-train_X[,select_feature]
X_logit_test<-test_data[,select_feature]
liblinear_fit<-LiblineaR(train_X, target = train_Y, type=0)
summary(liblinear_fit)

liblinear_fit$Type

pred<-predict(liblinear_fit, X_logit_test)
Reg_logit_fit<-pred$predictions
length(Reg_logit_fit)
################################

count_vec<-c()
vec_1<-as.numeric()
vec_2<-as.numeric()
vec_3<-as.numeric()
vec_4<-as.numeric()
count<-0
for(i in 1:length(Reg_logit_fit)){
  count_vec<-c(model_1[i],model_2[i,2],model_3[i,2], Reg_logit_fit[i])
  count_sort<-sort(table(count_vec),decreasing = TRUE)
  if(count_sort[1]!=4){
    vec_1<-append(vec_1, model_1[i])
    vec_2<-append(vec_2, model_2[i,2])
    vec_3<-append(vec_3, model_3[i,2])
    vec_4<-append(vec_4, Reg_logit_fit[i])
  }
  

  
}


compare_table<-cbind(vec_1,vec_2,vec_3,vec_4)
head(compare_table)
nrow(compare_table)
count


###Final combination of 4 models
final_result<-c()
###Get the final result
for(i in 1:length(Reg_logit_fit)){
  count_vec<-c(model_1[i],model_2[i,2],model_3[i,2], Reg_logit_fit[i])
  count_sort<-sort(table(count_vec),decreasing = TRUE)
  final_result[i]<-names(count_sort)[1]
}
final_result<-as.integer(final_result)

head(final_result)

####################################
count<-0
for(i in 1:length(Reg_logit_fit)){
  if(model_3[i,2]!=model_1[i])
    count=count+1
  
  
  
}
count


122/3360*100
predictions<-final_result

tibble(Id = seq_along(predictions), Prediction = predictions) %>%
  write_csv("4_model_ave_combination.csv")

count<-0
fixed_predictions<-predictions
for(i in 2:3926){
  if(fixed_predictions[i-1]==fixed_predictions[i+1]  &   fixed_predictions[i] !=fixed_predictions[i-1] ){
    count=count+1
    fixed_predictions[i]<-fixed_predictions[i-1]
  }
  
  
}
count
predictions<-fixed_predictions
predictions
tibble(Id = seq_along(predictions), Prediction = predictions) %>%
  write_csv("4_model_ave_combination_RoundedVersion.csv")




fixed_predictions
count<-0
i<-10
for(i in 4:3924){
  if(fixed_predictions[i-1]==fixed_predictions[i]){
    if(fixed_predictions[i-3]==fixed_predictions[i-2] & fixed_predictions[i+1]==fixed_predictions[i+2] & 
       fixed_predictions[i-2]==fixed_predictions[i+1] &  fixed_predictions[i]!=fixed_predictions[i+1]){
      count=count+2#mis-classification
    }
    
  }
}
count










####generate result
ave_model<-as.numeric()

for(i in 1:dim(model_1)[1]){
  if( model_1[i,2]==model_2[i,2])
    ave_model[i]<-model_1[i,2]
  else if( model_1[i,2]==model_3[i,2])
    ave_model[i]<-model_1[i,2]
  else if(model_2[i,2]==model_3[i,2])
    ave_model[i]<-model_2[i,2]
  else{
    choose_set<-c(model_1[i,2], model_2[i,2], model_3[i,2] )
    index_0<-sample(c(1,2,3),size=1)
    ave_model[i]<-choose_set[index_0]
  }
}
