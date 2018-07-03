library(class)
library(randomForest)
library(ROCR)
library(caret)
# data <- read.csv('http://christianherta.de/lehre/dataScience/machineLearning/data/titanic-train.csv',header=T,stringsAsFactors = F)
# keeps <- c("Survived", "Pclass", "Sex","Age","Fare")
# dataset <- data[keeps]
# dataset<-dataset[complete.cases(dataset),]
# sex <- ifelse(dataset$Sex=="male", 1, 0)
# dataset[,"Sex"] <- sex
# set.seed(12345) # Set Seed so that same sample can be reproduced in future also
# Now Selecting 80% of data as sample from total 'n' rows of the data
# sample <-sample(714,156,replace=FALSE)
# train.dataset <- dataset[-sample,]
# test.dataset <- dataset[sample,]
# write.csv(test.dataset, file="split-test.csv")
# write.csv(train.dataset, file="split-train.csv")
set.seed(43)
train.dataset <- read.csv('train.csv')
test.dataset <- read.csv('test.csv')
fit <- randomForest(as.factor(Survived) ~ Pclass + Sex + Age + Fare, data=train.dataset, importance=TRUE, ntree=5, maxnodes=16, nodesize=16)
Prediction <- predict(fit, test.dataset)

Prediction <- as.numeric(levels(Prediction)[as.integer(Prediction)])
rf_auc_1<-prediction(as.numeric(Prediction),test.dataset$Survived)
rf_prf<-performance(rf_auc_1, measure="tpr", x.measure="fpr")
rf_slot_fp<-slot(rf_auc_1,"fp")
rf_slot_tp<-slot(rf_auc_1,"tp")

rf_t <- table(test.dataset$Survived, Prediction)

# Model Accuracy
testError <- 1- (rf_t[1, 1] + rf_t[2, 2]) / sum(rf_t)



rf_perf_AUC=performance(rf_auc_1,"auc")
rf_AUC=rf_perf_AUC@y.values[[1]]


confusionMatrix <- confusionMatrix(as.numeric(Prediction),test.dataset$Survived)

# Print accuracy
cat("Test Error: ", testError)

# Print AUC
cat("AUC: ", rf_AUC)
print(confusionMatrix)
X11()
plot(rf_prf, col=rainbow(7), main="ROC curve Titanic (Random Forest)",
     xlab="FPR", ylab="TPR")
text(0.5,0.5,paste("AUC=",format(rf_perf_AUC@y.values[[1]],digits=5, scientific=FALSE)))


