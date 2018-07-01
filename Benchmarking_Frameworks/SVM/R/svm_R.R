#Big data 
#Jiwei Yu, 11/08/2017

#To clean up the memory of current R session
rm(list=ls(all=TRUE))
options(digits=3)

#get the data set and select the necessary columns 
titanic <- read.csv("http://christianherta.de/lehre/dataScience/machineLearning/data/titanic-train.csv",header=T)
sm_titanic_3<-titanic[,c(1,2,3,5,6,10)]

#remove the rows contains NA value
sm_titanic_3<-sm_titanic_3[complete.cases(sm_titanic_3),]
set.seed(43)

#seperate train data and test data
tst_idx<-sample(714,200,replace=FALSE)
tstdata<-sm_titanic_3[tst_idx,]
trdata<-sm_titanic_3[-tst_idx,]

#library for SVM
library("e1071")

#using train data set to predict 
attach(trdata)
model <- svm(Survived~Pclass+Sex+Age+Fare, data=trdata)
summary(model)
predict_svm <- predict(model,tstdata,type="class")

#predict_svm <- fitted(model)
result <- data.frame(PassengerId = tstdata$PassengerId, Survived = predict_svm)

# install.packages("caret")
require(caret)

#get the confusion matrix for result
confusionMatrix(as.numeric(predict_svm > 0.5),tstdata$Survived)

#compute AUC
require(ROCR)
predict_svm_rocr <- prediction(as.numeric(predict_svm),tstdata$Survived)
predict_prf_rocr <- performance(predict_svm_rocr,measure="tpr",x.measure="fpr")
predict_slot_fp <- slot(predict_svm_rocr,"fp")
predict_slot_tp <- slot(predict_svm_rocr,"tp")

predict_fpr3 <- unlist(predict_slot_fp)/unlist(slot(predict_svm_rocr,"n.neg"))
predict_tpr3 <- unlist(predict_slot_tp)/unlist(slot(predict_svm_rocr,"n.pos"))

predict_perf_AUC<- performance(predict_svm_rocr,"auc")
AUC = predict_perf_AUC@y.values[[1]]
print(AUC)
#library(MASS)
#library(ggplot2)
#library(dplyr)
#plot(model,trdata)
#plot(model, trdata, Survived~Sex, slice=list(Sepal.Width=3, Sepal.Length=4))
