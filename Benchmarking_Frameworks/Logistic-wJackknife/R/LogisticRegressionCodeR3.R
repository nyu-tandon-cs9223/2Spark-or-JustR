library(caret)
library(ggplot2)
library(lattice)
library(ROCR)
library(AUC)
training.data.raw <- read.csv('data.csv',header=T,na.strings=c(""))
data <- subset(training.data.raw,select=c(2,3,5,6,7,8,10))
newtimestamp <- Sys.time()
set.seed(as.numeric(newtimestamp))
data <- data[-c(as.numeric(newtimestamp)%%714)]
tstidx <- sample(713, 200, replace = FALSE)
test <- data[tstidx,]
training <- data[-tstidx,]
model <- glm(Survived ~.,family=binomial(link='logit'),data=training)
summary(model)
model_predict <- predict(model,test[, 2:7],type='response')
model_predict1 <- ifelse(model_predict > 0.5,1,0)
misClasificError <- mean(model_predict1 != test$Survived)
print(paste('Accuracy',1-misClasificError))
confusionMatrix(model_predict1,test[, 1])
pred <- prediction(model_predict, test$Survived)
auc <- performance(pred, measure = "auc")
auc <- auc@y.values[[1]]
print(paste('AUC', auc))