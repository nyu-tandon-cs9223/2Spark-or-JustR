# Benchmarking Supervised Learning Frameworks for Engineering Highly Scalable Intelligent Systems

## Instructions to run scripts:

Each folder has the two versions of the same model, one in R and one in Spark. 
The following instructions will help you execute the scripts. The instructions use SVM as an example.


### To run R scripts (from your shell):

1. Move to the respective folder:

	 `cd 2Spark-or-JustR/Benchmarking_Frameworks/SVM/R/`

2. Run script in R:

	`Rscript svm_R.R`

* Note: Make sure you have a working installation of R on your machine/instance.


### To run compile and run models in Spark (from your shell):

1. Move to the respective folder:

	 `cd 2Spark-or-JustR/Benchmarking_Frameworks/SVM/Spark/`

2. Test your mvn installation (optional):

	`mvn test`

3. If the compilation is successful, mvn is setup properly. Build the code into a jar file:

	`mvn package`

4. Submit the spark job for computation:

	`spark-submit --class Svm --master local[2] target/SVM-1.0.jar`

   where 
   
	--class should be the name of the class with the main function,
	
 	--master should be the mode for running spark. local[2] means run Spark in a standalone mode with two clusters. 'yarn' will run it in a cluster (make sure to have the yarn file ready).
	
	the third argument, target/SVM-1.0.jar is the path to the jar file itself.

* Note: The directory structure of each java model should be maintained. Any changes should be reflected in pom.xml.

* Make sure spark and scala have been added to the environment variables/$PATH.

## To run models from jar files:

1. Move to the respective folder:

	 `cd 2Spark-or-JustR/Benchmarking_Frameworks/SVM/Spark/`

2. Run the build script (every model with a Spark directory has one):

	`./build.script`
