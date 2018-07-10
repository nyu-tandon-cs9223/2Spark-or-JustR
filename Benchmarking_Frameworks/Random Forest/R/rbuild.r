library(party)
library(randomForest)
library(ROCR)
set.seed(43)
trdata <- read.csv("trdata.csv",header = TRUE)
tstdata <- read.csv("tstdata.csv",header = TRUE)
trdata$Survived <- as.factor(trdata$Survived)
trdata$Pclass <- as.factor(trdata$Pclass)
tstdata$Survived <- as.factor(tstdata$Survived)
tstdata$Pclass <- as.factor(tstdata$Pclass)
rf <- randomForest(Survived~.,data = trdata,ntree=200,test=tstdata)
rf.p = predict(rf,newdata=tstdata)
confusion_matrix <- table(tstdata$Survived,rf.p)
print("Confusion Matrix:")
confusion_matrix
accuracy <- (102+63)/nrow(tstdata)
print("Accuracy:")
accuracy
rf.pr = predict(rf,type="prob",newdata=tstdata)[,2]
rf.pred = prediction(rf.pr,tstdata$Survived)
auc <- performance(rf.pred,"auc")
auc <- unlist(slot(auc,"y.values"))
print("AUC:")
auc
