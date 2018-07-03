there are 2 classes in the jar file. 
LogisticRegression3 randomly split the dataset into two parts
command: spark-submit --master local[*] --class LogisticRegression3 ./spark-java-ml-1.0-SNAPSHOT.jar

LogisticRegression4 uses the dataset split in advance
command: spark-submit --master local[*] --class LogisticRegression4 ./spark-java-ml-1.0-SNAPSHOT.jar

there are also 2 Rscript file. 
LogisticRegressionCodeR3.R randomly split the dataset into two parts
command: Rscript LogisticRegressionCodeR3.R

LogisticRegressionCodeR4.R uses the dataset split in advance
command: Rscript LogisticRegressionCodeR4.R
