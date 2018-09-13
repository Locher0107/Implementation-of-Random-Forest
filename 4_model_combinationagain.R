


pred<-read.csv("4_model_ave_combination_Rounded_2times_FeatureSelect.csv")
pred<-pred[,2]
pred
xgb94<-read.csv("xgb939_select_feature.csv")
xgb94<-xgb94[,2]
xgb94
pred
feature_index<-read.csv("select_feature_index.csv")
feature_index<-feature_index[,1]



##
##record time
start<-proc.time()
accu<-as.numeric()
#library(foreach)
#library(doParallel)
library(e1071)
library(randomForest)
library(xgboost)
library(dplyr)
library(tidyverse)
library(tibble)
library(MASS)
#cl_2<-makeCluster(2)
#registerDoParallel(cl_2)


dat<-read.csv("train_human_python.csv", header=FALSE)
train_ID<-read.csv("training_subjects.csv",header=FALSE)
test_data<-read.csv("test_data.csv",header=FALSE)
train_label<-dat[,562]
train_X<-dat[,-562]

##splitdata
##SVM
###60 features maybe very important
##with cost=3 radial
accu<-c()

for(i in 1:18){
  X_train<-(train_X[,feature_index[1:60]])[train_ID[,1]!=i,]
  Y_train<-train_label[train_ID[,1]!=i]
  X_test<-(train_X[,feature_index[1:60]])[train_ID[,1]==i,]
  Y_test<-train_label[train_ID[,1]==i]
  Y_train<-as.factor(Y_train)
  
  msvm<-svm(X_train, Y_train,cost=3,kernel="radial")
  predictions<-predict(msvm, X_test)
  accu[i]<-mean(predictions==Y_test)
  cat(accu[i],"\n")
  
}
mean(accu)







##Test svm with 10-fold cross validation
########
accu<-c()

index<-list()
origin<-1:6373
##how to split data set
index[[1]]<-sample(1:6373,size=0.1*6373,replace = FALSE)

for(i in 2:10){
  origin<-origin[!origin%in%index[[i-1]]]
  index[[i]]<-sample(origin,size=0.1*6373,replace = FALSE  )
}


for(i in 1:10){
  X_train<-(train_X[-index[[i]],feature_index[1:60]])
  Y_train<-train_label[-index[[i]]]
  X_test<-(train_X[index[[i]],feature_index[1:60]])
  Y_test<-train_label[index[[i]]]
  Y_train<-as.factor(Y_train)
  
  msvm<-svm(X_train, Y_train,cost=3,kernel="radial")
  predictions<-predict(msvm, X_test)
  accu[i]<-mean(predictions==Y_test)
  cat(i," ",accu[i],"\n")
  
  
}
mean(accu)


#####Next we test random forest


accu<-c()

for(i in 1:18){
  X_train<-(train_X[,feature_index[1:300]])[train_ID[,1]!=i,]
  Y_train<-train_label[train_ID[,1]!=i]
  X_test<-(train_X[,feature_index[1:300]])[train_ID[,1]==i,]
  Y_test<-train_label[train_ID[,1]==i]
  Y_train<-as.factor(Y_train)
  
  RF_fit<-randomForest(x=X_train,y=Y_train,ntree = 1000,nodesize = 3)
  RF_pred<-predict(RF_fit, newdata = X_test,type="class")
  mean(RF_pred==Y_test)
  accu[i]<-mean(RF_pred==Y_test)
  cat(accu[i],"\n")
  
}
mean(accu)





X_train<-train_X[,feature_index[1:60]]
Y_train<-as.factor(train_label)
X_test<-test_data[,feature_index[1:60]]
msvm<-svm(X_train, Y_train,cost=3,kernel="radial")
predictions<-predict(msvm, X_test)
colnames(predictions)<-NULL
as.numeric(predictions[1:1000])


RF_fit<-randomForest(x=X_train,y=Y_train,ntree = 1000,nodesize = 5)
RF_pred<-predict(RF_fit, newdata = X_test,type="class")
as.numeric(RF_pred)
tibble(Id = seq_along(predictions), Prediction = RF_pred) %>%
  write_csv("RF_select.csv")





###next I will use 30 svm model to fit 

predlist<-list()
for(i in 1:30){
  
  index<-sample(1:6373,size=0.9*6373,replace = FALSE)
  
  X_train<-(train_X[index,feature_index[1:60]])
  Y_train<-train_label[index]
  Y_train<-as.factor(Y_train)
  test_data<-as.matrix(test_data)
  X_test<-test_data[,feature_index[1:60]]


  msvm<-svm(X_train, Y_train,cost=3,kernel="radial")
  predlist[[i]]<-predict(msvm, X_test)

  cat(i,"\n")
}
svm_940<-matrix(unlist(predlist),ncol=30,byrow = FALSE)
svm_940
dim(svm_940)
count<-0
mat<-as.numeric(mat)
mat<-c()
for(i in 1:3926){
  t<-sort(table(svm_940[i,]),decreasing = TRUE)[1]
  if(t!=30){
    mat<-rbind(mat,svm_940[i,])
    count=count+1
  }
  
}
mat
dim(mat)
index_6
class(mat)

count<-0
mat[90:120,]
mat<-mapply(mat,FUN = as.numeric)
mat<-matrix(mat,nrow=188,ncol=30)
mat
class(mat)
count<-0
index_6<-c()
for(i in 1:188){
  for(j in 1:30){
    if(mat[i,j]==6){
      count=count+1
      index_6[count]<-i
      break
    }
      
  }
  
}
index_6
mat[c(70,111),]
#####only 2 point include 6
mat

yty<-as.matrix(table(mat[1,]))
yty


count<-0
xgb94
for(i in 1:3926){
  if(xgb94[i]==6 & svm_940_predict[i]!=6)
    count<-count+1
}
count
count<-0
for(i in 1:188){
  if(4 %in% mat[i,]){
    if((table(mat[i,])["5"]>=5) & (table(mat[i,])["4"]>=5) ){
      count=count+1
    }
    
  }
    
}



count


table(mat[5,])
mat

count

mat<-as.numeric(mat)
(6 %in%mat)



count
svm_940_predict<-c()
for(i in 1:3926){
  label<-names(sort(table(svm_940[i,]), decreasing = TRUE  ))[1]
  svm_940_predict[i]<-as.integer(label)
  
}
svm_940_predict[1000:2000]
pred[1:145]

warnings()
tibble(Id = seq_along(predictions), Prediction = svm_940_predict) %>%
  write_csv("svm940_predict_selFeature.csv")




###fit group A
###TEST THE GROUP A model do binary classification
index_45_train<-train_label==4 | train_label==5

X_group_A_train<-train_X[index_45,]
Y_group_A_train<-train_label[index_45]

group_A_ID<-train_ID[index_45_train,1]

index_45<-svm_940_predict==4 | svm_940_predict==5
X_group_A_test<-test_data[index_45,]


X_group_A_test<-as.data.frame(X_group_A_test)
index_45<-as.data.frame(index_45)
write_csv(cbind(X_group_A_train,group_A_ID,Y_group_A_train),"X_group_A_train.csv",col_names =FALSE)
write_csv(X_group_A_test,"X_group_A_test.csv",col_names =FALSE)
write_csv(index_45,"svm_index45.csv",col_names =FALSE)
head(index_45)




value<-c()
for(mm in 50:100){

mm=45
accu<-c()
for(i in 1:18){
  X_train<-X_group_A_train[group_A_ID!=i,feature_index[1:mm]]
  Y_train<-Y_group_A_train[group_A_ID!=i]
  X_test<-X_group_A_train[group_A_ID==i,feature_index[1:mm]]
  Y_test<-Y_group_A_train[group_A_ID==i]
  

  
  ##Support Vector Machine
  #msvm<-svm(X_train, Y_train,cost=2,kernel="radial")
  #predictions<-predict(msvm, X_test)
  
  #Random Forest
  #RF_fit<-randomForest(x=X_train,y=Y_train,ntree = 500,nodesize = 3)
  #RF_pred<-predict(RF_fit, newdata = X_test,type="class")
  
  #LDA
  #LDA_fit<-lda(x=X_train, grouping=Y_train)
  #LDA_predictions<-predict(LDA_fit,X_test)$class
  
  ##Logistic
  X_train<-as.matrix(X_train)
  X_test<-as.matrix(X_test)
  X_test<-as.data.frame(X_test)
  
  

  Y_train<-Y_train-4

  Y_test<-Y_test-4
  Y_train<-as.numeric(Y_train)
  
  data_logit<-as.data.frame(cbind(X_train,Y_train))
  model<-glm(Y_train~., data=data_logit,family=binomial(link = "logit"))
  summary(model)
  log_predict<-predict(model,X_test,type="response")
  log_predict<-ifelse(log_predict>0.5,1,0)
  accu[i]<-mean(log_predict==Y_test)
  cat(accu[i],"\n")
}

  mean(accu)
  value<-c(value,mean(accu))
}


plot(value)






length(log_predict)





dim(train_X)
dim(X_train)
dim(X_group_A_train)



