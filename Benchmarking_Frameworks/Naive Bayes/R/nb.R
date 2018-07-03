library(caret)
library(e1071)
library(pROC)
require(ROCR)
titanic <- read.csv("./titanic-train.csv",header=T)
head(titanic)
titanic$Ticket <- NULL
titanic$Name <- NULL
titanic[is.na(titanic)]<-0
titanic$Cabin <- as.numeric(titanic$Cabin)
titanic$Sex <- as.numeric(titanic$Sex)
titanic$Embarked <- as.numeric(titanic$Embarked)
write.table(titanic, file="./Mydata.csv" ,col.names=FALSE , row.names = FALSE)
set.seed(43)
test_index <- sample(714,200,replace=FALSE)
tstdata = titanic[test_index,]
trdata = titanic[-test_index,]
write.table(tstdata, file="./test.csv" ,col.names=FALSE , row.names = FALSE)
write.table(trdata, file="./train.csv" ,col.names=FALSE , row.names = FALSE)
model <- naiveBayes(as.factor(Survived)~.,data=trdata)
model
preds <- predict(model,tstdata[,3:9],prob=TRUE)
head(preds)
pr <- prediction(as.numeric(preds),as.numeric(tstdata[,2]))
prf <- performance(pr,"tpr","fpr")
plot(prf)
prf
auc <- performance(pr,'auc')
auc
auc_final <- auc@y.values[[1]]
text(0.4,0.6,paste("NAIVE BAYES AUC=",auc_final))
confusionMatrix(preds,tstdata[,2])
# CM 0    1
# 0 102 37
# 1  12  49
#AUC = 0.725
