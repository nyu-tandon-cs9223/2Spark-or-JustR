library(e1071)
library(ROCR)

#setwd('~/Spark_Project')
#You can set your own directory.

df = read.table('titanic-train.csv', header = TRUE, sep = ',')
head(df)
df2 = df[ , -which(names(df) %in% c("PassengerId","Name","SibSp","Ticket","Cabin","Embarked","Parch"))]
head(df2)
df3 = df2[complete.cases(df2),]
head(df3)

set.seed(43)
tst_idx = sample(714, 200, replace = FALSE)
test = df3[tst_idx,]
training = df3[-tst_idx,]

X_training = training[, -which(names(training) %in% c("Survived"))]
y_training = training[,c("Survived")]
X_test = test[, -which(names(test) %in% c("Survived"))]
y_test = test[,c("Survived")]


aucs = c()  
plot(x = NA, y = NA, xlim = c(0, 1), ylim = c(0, 1), ylab = 'True Positive Rate', xlab = 'False Positive Rate', bty = 'n')

lvls = c(0, 1)  
for (type.id in 1:2) {
  type = as.factor(training$Survived == lvls[type.id])
  model = model <- svm(as.formula(paste("type", "~ .", sep = " ")), data = training[,-1], kernel = "linear", probability = T, type = "C-classification",cost = 1)
  #Survived = predict(model, test[,-1])
  #score <- c()
  #score[Survived == TRUE] <- 1
  #score[Survived == FALSE] <- -1
  
  Survived = predict(model, test[,-1], decision.values = T, probability = T)
  score = attr(Survived, "probabilities")[,2]
  
  actual.Survived = test$Survived == lvls[type.id]
  pred = prediction(score, actual.Survived)
  perf = performance(pred, "tpr", "fpr")
  roc.x = unlist(perf@x.values)
  roc.y = unlist(perf@y.values)
  lines(roc.y ~ roc.x, col = type.id + 1, lwd = 2)
  nauc = performance(pred, "auc")
  nauc = unlist(slot(nauc, "y.values"))
  aucs[type.id] = nauc
}
legend("bottomright", legend = c("1 vs others", "2 vs others"), col = 2:3, lty = 1, lwd = 2)
lines(x = c(0, 1), c(0, 1))
cat("AUC", fill = T)
aucs
