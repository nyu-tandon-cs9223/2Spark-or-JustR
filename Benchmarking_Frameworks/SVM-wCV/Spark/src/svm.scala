/**
  * Created by yatingyu on 11/20/17.
  */

import org.apache.spark.ml.classification.LinearSVC
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}
import org.apache.spark.sql.SparkSession

object svm {
  val spark: SparkSession = SparkSession.builder.master("local[2]").getOrCreate

  case class Initial(survived: Option[Int], pclass: Option[Int], sex: Option[String], age: Option[Double], fare: Option[Double])
  case class Final(survived: Double, pclass: Double, sex: Double, age: Double, fare: Double)

  def mapData(in: Initial) = Final(
    in.survived.map(_.toDouble).getOrElse(0),
    in.pclass.map(_.toDouble).getOrElse(0),
    in.sex match { case Some("female") => 1; case Some("male") => 2; case _ => 0 },
    in.age.getOrElse(0),
    in.fare.getOrElse(0)
  )

  def main(args: Array[String]): Unit = {
    println("Computing......")
    println("May take 10 or more seconds......")

    val file1 = spark.read.option("header", "true").option("mode", "DROPMALFORMED")
      .option("inferSchema", "true").csv("./resources/train.csv")
    val file2 = spark.read.option("header", "true").option("mode", "DROPMALFORMED")
      .option("inferSchema", "true").csv("./resources/test.csv")
    val assembler = new VectorAssembler()
      .setInputCols(Array("pclass", "sex", "age", "fare"))
      .setOutputCol("features")

    import spark.implicits._
    val data1 = assembler.transform(file1.as[Initial].map(mapData))
    val data2 = assembler.transform(file2.as[Initial].map(mapData))
    val training = data1.drop("pclass", "sex", "age", "fare").toDF("label", "features")
    val test = data2.drop("pclass", "sex", "age", "fare").toDF("label", "features")

    // Linear SVM model
    val lsvc = new LinearSVC()
      .setMaxIter(100)
      .setTol(1e-6)
      .setStandardization(true)
      .setFitIntercept(true)
      .setThreshold(0.0)
      .setLabelCol("label")
      .setFeaturesCol("features")

    //Parameter C
    val paramGrid = new ParamGridBuilder()
      .addGrid(lsvc.regParam, Array(0.005,0.05,0.1))
      .build()

    // Evaluate accuracy at each iteration
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")

    // Cross Validation for lsvc model
    val cv = new CrossValidator()
      .setEstimator(lsvc)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(5)

    // Train on whole training set use best model
    val cvModel = cv.fit(training)
    println("1. 5-fold cross validation for C:")
    println("Average accuracy for each parameter(C):")
    cvModel.avgMetrics.foreach(println)
    println("\nBest Model Parameters:")
    println(cvModel.bestModel.extractParamMap())

    // Test on the test set
    val predictions = cvModel.transform(test)

    val accuracy = evaluator evaluate predictions

    val predictionAndLabels = predictions.select($"prediction", $"label").rdd
      .map(row => (row.toSeq.toArray.head.toString().toDouble, row.toSeq.toArray.last.toString().toDouble))

    println("\n2. Test on the test set:")
    println("accuracy = " + accuracy)
    println("corrects  = " + predictionAndLabels.filter(r => r._1 == r._2).count.toDouble)
    val matrix = new BinaryClassificationMetrics(predictionAndLabels)
    val auROC = matrix.areaUnderROC()
    println("Area under ROC = " + auROC)

    val metrics = new MulticlassMetrics(predictionAndLabels)
    println("Confusion matrix:")
    println(metrics.confusionMatrix)

  }
}
