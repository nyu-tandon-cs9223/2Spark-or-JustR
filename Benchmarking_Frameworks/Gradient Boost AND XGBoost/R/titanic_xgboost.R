#install.packages("caret", dependencies=TRUE)

library(xgboost)
library(caret)
library(rpart)

set.seed(43)

#Load the csv file 
titanic <- read.csv("http://christianherta.de/lehre/dataScience/machineLearning/data/titanic-train.csv",header=T)

#filter out the necessary colum
sm_titanic <- titanic[,c(2,3,5,6,10)]
#filter the column with missing value
sm_titanic <- sm_titanic[complete.cases(sm_titanic),]

#Convert the sex to number: male: 1, female: 2
sm_titanic$Sex <- as.numeric(sm_titanic$Sex) - 1

sm_titanic <- as.matrix(sm_titanic)

#divide the dataset into train set and test set
tst_index <- sample(714, 200, replace = FALSE)

test <- sm_titanic[tst_index,]
train <- sm_titanic[-tst_index,]

write.csv(train, file = "~/titanic_train.csv", quote = FALSE)
write.csv(test, file = "~/titanic_test.csv", quote = FALSE)

# Using the cross validation to estimate our error rate:
param <- list("objective" = "binary:logistic", "max_depth" = 5)

cv.nround <- 100
cv.nfold <- 2
xgboost_cv = xgb.cv(param=param, data = train[, -c(1)], label = train[, c(1)], nfold = cv.nfold, nrounds = cv.nround)

# Fitting with the xgboost model
nround  = 5
fit_xgboost <- xgboost(param =param, data = train[, -c(1)], label = train[, c(1)], nrounds=nround)

# Get the feature real names
names <- dimnames(train)[[2]]

# Compute feature importance matrix
importance_matrix <- xgb.importance(names, model = fit_xgboost)

# Plotting the importance of the martix
xgb.plot.importance(importance_matrix)

# Prediction on test and train sets
pred_xgboost_test <- predict(fit_xgboost, test[, -c(1)])
pred_xgboost_train <- predict(fit_xgboost, train[, -c(1)])

# Since xgboost gives a survival probability prediction, we need to find the best cut-off:
proportion <- sapply(seq(.3,.7,.01),function(step) c(step,sum(ifelse(pred_xgboost_train<step,0,1)!=train[, c(1)])))
dim(proportion)
# Applying the best cut-off on the train set prediction for score checking
#predict_xgboost_train <- ifelse(pred_xgboost_train<proportion[,which.min(proportion[2,])][1],0,1)
# head(predict_xgboost_train)
# score <- sum(train[, c(1)] == predict_xgboost_train)/nrow(train)
# score

# Applying the best cut-off on the test set
predict_xgboost_test <- ifelse(pred_xgboost_test<proportion[,which.min(proportion[2,])][1],0,1)
predict_xgboost_test <- as.data.frame(predict_xgboost_test) # Conveting the matrix into a dataframe
predict_xgboost_test

#compare the result
score <- sum(test[, c(1)] == predict_xgboost_test)/nrow(test)
score

#result output evaluation
test <- as.data.frame(test)
require(ROCR)
xgb_auc<-prediction(predict_xgboost_test, test$Survived)
xgb_prf<-performance(xgb_auc, measure = "tpr", x.measure = "fpr")
xgb_slot_fp <- slot(xgb_auc, "fp")
xgb_slot_tp <-slot(xgb_auc, "tp")

xgb_fpr<-unlist(xgb_slot_fp)/unlist(slot(xgb_auc,"n.neg"))
xgb_tpr<-unlist(xgb_slot_tp)/unlist(slot(xgb_auc, "n.pos"))
xgb_pref_AUC = performance(xgb_auc, "auc")
xgb_AUC = xgb_pref_AUC@y.values[[1]]
"AUC"
xgb_AUC
plot(xgb_prf, main = "ROC plot", xlab = "FPR", ylab = "TPR")
#this need extra library and package installment
result <- confusionMatrix(as.numeric(unlist(predict_xgboost_test)), as.numeric(test$Survived))
result
