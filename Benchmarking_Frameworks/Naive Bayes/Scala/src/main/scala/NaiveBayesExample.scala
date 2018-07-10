import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib._
import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.rdd.RDD



object NaiveBayesExample {

  def main(args:Array[String]) : Unit = {
    val sc = SparkContext.getOrCreate()

    val datapre = sc.textFile("Mydata.csv")
    val parseddatapre = datapre.map { line =>
      val parts = line.split(',')
      LabeledPoint(parts(1).toDouble, Vectors.dense(parts(0).toDouble, parts(2).toDouble, parts(3).toDouble, parts(4).toDouble, parts(5).toDouble, parts(6).toDouble, parts(7).toDouble, parts(8).toDouble))
    }


    val splits = parseddatapre.randomSplit(Array(0.6, 0.4), seed = 11L)
    val training = splits(0)
    val test = splits(1)
    val model = NaiveBayes.train(training)
    val predictionAndLabel = test.map(p => (model.predict(p.features), p.label))
    val accuracy = 1.0 * predictionAndLabel.filter(x => x._1 == x._2).count() / test.count()

    println("the accuracy is " + accuracy)

    val metrics = new BinaryClassificationMetrics(predictionAndLabel)
    val roc1 = metrics.roc

    println("the ROC is " + roc1.collect())

    val auc1 = metrics.areaUnderROC

    println("the AUC is " + auc1)

    val metrics2 = new MulticlassMetrics(predictionAndLabel)
    val confmatrix = metrics2.confusionMatrix
    println("**************************************************OUTPUT*******************************************")
    println("the accuracy is " + accuracy)
    println("the ROC is " + roc1.collect())
    println("the AUC is " + auc1)
    println("the confusionmatrix is  " + confmatrix)
    println("**************************************************OUTPUT*******************************************")

  }

}
