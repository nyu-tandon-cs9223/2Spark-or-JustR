#install.packages("gbm", dependencies=TRUE)
#install.packages("caret", dependencies=TRUE)
library(gbm)
library(caret)
library(rpart)

set.seed(43)

#Load the csv file 
titanic <- read.csv("http://christianherta.de/lehre/dataScience/machineLearning/data/titanic-train.csv",header=T)

#filter out the necessary colum
sm_titanic <- titanic[,c(2,3,5,6,10)]
#filter the column with missing value
sm_titanic <- sm_titanic[complete.cases(sm_titanic),]

#Convert the sex to number: male: 1, female: 0
sm_titanic$Sex <- as.numeric(sm_titanic$Sex) - 1

#divide the dataset into train set and test set
tst_index <- sample(714, 200, replace = FALSE)

test <- sm_titanic[tst_index,]
train <- sm_titanic[-tst_index,]

write.csv(train, file = "~/titanic_train.csv", quote = FALSE)
write.csv(test, file = "~/titanic_test.csv", quote = FALSE)

#https://www.r-bloggers.com/predicting-titanic-deaths-on-kaggle-ii-gbm/
gbm_fit <- gbm(Survived ~ ., 
               n.cores = 2,
               interaction.depth=5,
               data=train)
pred_test <- predict(gbm_fit, test[, -c(1)], n.trees = 100, single.tree = TRUE, type = "response")
pred_train <- predict(gbm_fit, train[, -c(1)], n.trees = 100, single.tree = TRUE, type = "response")
# Since gbm gives a survival probability prediction, we need to find the best cut-off
proportion <- sapply(seq(.3,.7,.01),function(step) c(step,sum(ifelse(pred_train<step,0,1)!=train[, c(1)])))
dim(proportion)
# Applying the best cut-off on the train set prediction for score checking
# predict_train <- ifelse(pred_train<proportion[,which.min(proportion[2,])][1],0,1)
# head(predict_train)
# score <- sum(train[, c(1)] == predict_xgboost_train)/nrow(train)
# score

predict_test <- ifelse(pred_test<proportion[,which.min(proportion[2,])][1],0,1)
predict_test
# check the accuracy of the prediction
score <- sum(test[, c(1)] == predict_test)/nrow(test)
score

# evaluation the result
require(ROCR)
gbm_auc<-prediction(predict_test, test$Survived)
gbm_prf<-performance(gbm_auc, measure = "tpr", x.measure = "fpr")
gbm_slot_fp <- slot(gbm_auc, "fp")
gbm_slot_tp <-slot(gbm_auc, "tp")

gbm_fpr<-unlist(gbm_slot_fp)/unlist(slot(gbm_auc,"n.neg"))
gbm_tpr<-unlist(gbm_slot_tp)/unlist(slot(gbm_auc, "n.pos"))
gbm_pref_AUC = performance(gbm_auc, "auc")
gbm_AUC = gbm_pref_AUC@y.values[[1]]
"AUC"
gbm_AUC

plot(gbm_prf, main = "ROC plot", xlab = "FPR", ylab = "TPR")
#this need extra library and package installment
result <- confusionMatrix(as.numeric(unlist(predict_test)), as.numeric(test$Survived))
sprintf("Accuracy: %f", score)
print(result)
