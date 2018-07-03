#install.packages("caret")
library(caret)
library(rpart)
library(gbm)
titanic <- read.csv("titanic-train.csv",header=T,stringsAsFactors = F)
#titanic <- read.csv("/Users/yuejing/Downloads/titanic-train.csv",header=T,stringsAsFactors = F)
data <- titanic[complete.cases(titanic),]
set.seed(43)
test_id <- sample(714, 200, replace=FALSE)
titanic_testdata <- data[test_id,]
titanic_traindata <- data[-test_id,]
write.csv(titanic_testdata, "titanic_test_R.csv")
write.csv(titanic_traindata, "titanic_train_R.csv")

titanic_train <- read.csv("titanic_train_R.csv",header=T,stringsAsFactors = F)
titanic_test <- read.csv("titanic_test_R.csv",header=T,stringsAsFactors = F)

traindata <- titanic_train[,c(3,4,6,7,11)]
trainsex <- ifelse(traindata$Sex=="male", 1, 0) 
traindata[,"Sex"] <- trainsex

testdata <- titanic_test[,c(3,4,6,7,11)]
testsex <- ifelse(testdata$Sex=="male", 1, 0) 
testdata[,"Sex"] <- testsex

gbm_data <- gbm(Survived ~ ., cv.folds=11,n.cores=2,interaction.depth=5,shrinkage = 0.0005,distribution="adaboost",data=traindata,n.trees=10000)
gbm_pred <- predict(gbm_data,testdata[,2:5], type="response")
sol <- as.numeric(gbm_pred > 0.5)
#normPreds <- (gbm_pred-min(gbm_pred))/(max(gbm_pred)-min(gbm_pred))
#sol <- testdata
#sol$Survived <- as.numeric(normPreds > 0.5)
#same <- ifelse(sol$Survived == testdata$Survived, 0, 1)
#percentage = sum(same == 0)/200
result <- confusionMatrix(sol, testdata$Survived)
print(result)

#compute AUC
require(ROCR)
gbm_auc <- prediction(as.numeric(gbm_pred > 0.5), testdata$Survived)
gbm_prf <- performance(gbm_auc, measure="tpr", x.measure = "fpr")
gbm_perf_AUC = performance(gbm_auc, "auc")
gbm_AUC_y = gbm_perf_AUC@y.values[[1]]
print(gbm_AUC_y)
#gbm_slot_fp <- slot(gbm_auc, "fp")
#gbm_slot_tp <- slot(gbm_auc, "tp")
#gbm_fpr <- unlist(gbm_slot_fp)/unlist(slot(gbm_auc, "n.neg"))
#gbm_tpr <- unlist(gbm_slot_tp)/unlist(slot(gbm_auc, "n.pos"))
plot(gbm_prf, main = "ROC plot", xlab = "FPR", ylab = "TPR")
text(0.4,0.6,paste("GBM AUC",format(gbm_AUC_y,digits = 5, scientific = FALSE)))
