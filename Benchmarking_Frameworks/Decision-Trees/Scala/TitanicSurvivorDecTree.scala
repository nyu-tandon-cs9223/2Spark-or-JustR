/**
  * Created by Shweta on 12/7/2017.
  */
package ScalaProject
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.sql.{Row, SQLContext, SparkSession}
import Stats.confusionMatrix

object TitanicSurvivorDecTree {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder().appName("Classification of Titanic Survivors with DecisionTree").master("local[*]").getOrCreate()
    import spark.implicits._
    val sc = spark.sparkContext
    val sqlContext = new SQLContext(sc)
    val tr_path = "titanic-train.csv"
    val ts_path = "titanic-test.csv"
    val df_train = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").load(tr_path)
    val df_test = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").load(ts_path)

    val train_rdd = df_train.rdd.map(row => {
      var age = row.get(5)
      if (age == null) {
        age = "30"
      }
      var emb = row.getString(11)
      if (emb == null) {
        emb = "C"
      }
      var fare =row.get(9)
      if (fare == null){
        fare = "151.5"
      }
      var parch = row.get(7)
      if (parch == null){
        parch ="0"
      }
      row.get(0) + "," +
      row.get(1)+ "," +
      row.get(2)+ "," +
      row.getString(4)+ "," +
      age+ "," +
      row.get(6)+ "," +
      parch + "," +
      fare + "," +
      emb
    })

    val test_rdd = df_test.rdd.map(row => {
      println(row)
      var age = row.get(5)
      if (age == null) {
        age = "30"
      }
      var emb = row.getString(11)
      if (emb == null) {
        emb = "C"
      }
      var fare =row.get(9)
      if (fare == null){
        fare = "151.5"
      }
      var parch = row.get(7)
      if (parch == null){
        parch ="0"
      }
      row.get(0) + "," +
      row.get(1)+ "," +
      row.get(2)+ "," +
      row.getString(4)+ "," +
      age+ "," +
      row.get(6)+ "," +
      parch + "," +
      fare + "," +
      emb
    })

    val train_data = train_rdd.map { row =>
      TitanicSurvivorModel(row.split(",")).toLabeledPoint
    }

    val test_data = test_rdd.map { row =>
      TitanicSurvivorModel(row.split(",")).toLabeledPoint
    }

    val numClasses = 2
    val impurity = "entropy"
    val maxDepth = 20
    val maxBins = 34

    val dtree = DecisionTree.trainClassifier(train_data, numClasses, TitanicSurvivorModel.categoricalFeaturesInfo(), impurity, maxDepth, maxBins)

    test_data.foreach {
      x => println(s"Predicted: ${dtree.predict(x.features)}, Label: ${x.label}")
    }

    val predictionsAndLabels = test_data.map {
      point => (dtree.predict(point.features), point.label)
    }

    val stats = Stats(confusionMatrix(predictionsAndLabels))
    println("-------Stats-------")
    println(stats.toString)

    val metrics = new BinaryClassificationMetrics(predictionsAndLabels)
    val auROC = metrics.areaUnderROC
    println("---------auROC-------------")
    println(auROC)


    spark.stop()
  }

}
