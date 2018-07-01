rm(list=ls(all=TRUE))
options(digits=3)
# libraries needed for this program
library(tree)
#library (ISLR)
library(rpart)
library(rpart.plot)
library(MASS)
library(ggplot2)
library(dplyr)
require(ROCR) # use ROCR package
require(caret)
# load data
titanic <- read.csv("http://christianherta.de/lehre/dataScience/machineLearning/data/titanic-train.csv", header = T)
# only need 2, 3, 5, 6, 10 columns
sm_titantic_3 <-titanic[,c(1,2,3,5,6,10)]
# remove the rows contianin NA value
sm_titanic_3 <- sm_titantic_3[complete.cases(sm_titantic_3),]
set.seed(43)
tst_idx <- sample(714,200, replace = FALSE)
# test data and train data
test <- sm_titanic_3[tst_idx,]
train <- sm_titanic_3[-tst_idx,]
#write train and test data to directory
write.csv(test, file ="test.csv")
write.csv(train, file ="train.csv")
# data explore
summary(train)
# this shows the number of NAs for each variable in the dataset
sapply(train, function(x) sum(is.na(x)))
summary(train$Sex)
# personal preference - optional
attach(train)
# using decision tree() for predcition
DT <- rpart(Survived~Age+Fare+Pclass+Sex, data = train, method ="class")
#DT <- rpart(Survived~., data = train, method ="class", cp = 0)
summary(DT)
#Plot the CV error curve for the tree
#plotcp(DT)
printcp(DT)
# output
predict_dt <- predict(DT, test, type = "class")
#plot the CV error for the tree
result <- data.frame(PassengerID = test$PassengerId, Survived = predict_dt)
#compute AUC
predict_dt_rocr <- prediction(as.numeric(predict_dt), test$Survived)
predict_prf_rocr <- performance(predict_dt_rocr, measure="tpr", x.measure = "fpr")
predict_perf_AUC <- performance(predict_dt_rocr, "auc")
# plot it and visualize
plot(predict_prf_rocr, main ="ROC plot cp = 0.03(DTREE using rpart)")
text(0.5,0.5, paste("AUC=", format(predict_perf_AUC@y.values[[1]], digits = 5, scientific = FALSE)))
# number of true and false
confusionTable <- table(test$Survived, predict_dt)
#write the output to directory
write.csv(result, file ="result.csv", row.names = FALSE)
# confusion matrix of output
confusionMatrix(predict_dt, test$Survived, mode ="prec_recall")
