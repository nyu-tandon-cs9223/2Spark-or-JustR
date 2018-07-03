data <- read.csv('http://christianherta.de/lehre/dataScience/machineLearning/data/titanic-train.csv',header=T,stringsAsFactors = F)
keeps <- c("Survived", "Pclass", "Sex","Age","Fare")
dataset <- data[keeps]
dataset<-dataset[complete.cases(dataset),]
sex <- ifelse(dataset$Sex=="male", 1, 0)
dataset[,"Sex"] <- sex
set.seed(12345) # Set Seed so that same sample can be reproduced in future also
Now Selecting 80% of data as sample from total 'n' rows of the data
sample <-sample(714,156,replace=FALSE)
train.dataset <- dataset[-sample,]
test.dataset <- dataset[sample,]
write.csv(test.dataset, file="split-test.csv")
write.csv(train.dataset, file="split-train.csv")
