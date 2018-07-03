library(glmnet)
library(ROCR)

trainData<-read.csv("train.csv", header=FALSE,stringsAsFactors=FALSE)
testData<-read.csv("test.csv", header=FALSE,stringsAsFactors=FALSE)

testData$V1 <- factor(testData$V1) #survived
testData$V2 <- factor(testData$V2) #pclass
testData$V3 <- factor(testData$V3) #sex

trainData$V1 <- factor(trainData$V1) #survived
trainData$V2 <- factor(trainData$V2) #pclass
trainData$V3 <- factor(trainData$V3) #sex


x <- data.matrix(trainData[,c(2,3,4,5)])
newx <- data.matrix(testData[,c(2,3,4,5)])
y <- trainData$V1

cvfit.m.ridge <- cv.glmnet(x, y, 
                           family = "binomial", 
                           alpha = 0,
                           type.measure = "class")

pred_ridge <- predict(cvfit.m.ridge, newx = newx, s = 'lambda.min', type='response')
pred_ridge <- round(pred_ridge)
confusion.matrix <- prop.table(table(pred_ridge, testData$V1))
accuracy <- confusion.matrix[1,1] + confusion.matrix[2,2] 
paste("Accuracy: ", accuracy)


ridge_pred_class <- prediction(pred_ridge,testData$V1)
ridge_perf_rocr <- performance(ridge_pred_class, measure='tpr',x.measure='fpr')
ridge_slot_fp <- slot(ridge_pred_class, "fp")
ridge_slot_tp <- slot(ridge_pred_class, "tp")


ridge_perf_auc <- performance(ridge_pred_class, measure="auc")
ridge_AUC <- ridge_perf_auc@y.values[[1]]
paste("AUC: ", ridge_AUC)
