print("################### R #################")
require("e1071")
library("pROC")
traindata <- read.csv("train.csv", header = TRUE)
testdata <- read.csv("test.csv", header = TRUE)

# cross validation to evaluate parameter C on training set
sv<-svm(Survived ~ ., data=traindata, kernel='linear',type='C-classification',scale =TRUE,tol=1e-6,max_iter=100,cost=0.005,cross=5)
print("##### Summary of cost = 0.005:")
print(summary(sv))
sv<-svm(Survived ~ ., data=traindata, kernel='linear',type='C-classification',scale =TRUE,tol=1e-6,max_iter=100,cost=0.05,cross=5)
print("##### Summary of cost = 0.05:")
print(summary(sv))
sv<-svm(Survived ~ ., data=traindata, kernel='linear',type='C-classification',scale =TRUE,tol=1e-6,max_iter=100,cost=0.1,cross=5)
print("#####Summary of cost = 0.1:")
print(summary(sv))


# test on the test set
sv<-svm(Survived ~ ., data=traindata, kernel='linear',type='C-classification',scale =TRUE,tol=1e-6,max_iter=100,cost=0.1)
pre <- predict(sv,testdata)
label <- testdata$Survived

prediction <- as.numeric(as.character(pre))
roc_obj <- roc(label, prediction)

# AUC number and confusion matrix
print("Results:")
print(auc(roc_obj))
print("Confusion Matrix:")
print(table(label, pre))

