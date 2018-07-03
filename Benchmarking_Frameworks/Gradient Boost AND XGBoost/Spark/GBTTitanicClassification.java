/**
 * Created by Hunter6 on 11/30/17.
 *
 * The code is writen based on a spark java example:
 * https://spark.apache.org/docs/2.1.0/mllib-ensembles.html#gradient-boosted-trees-gbts
 *
 */

import org.apache.log4j.LogManager;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.GradientBoostedTrees;
import org.apache.spark.mllib.tree.configuration.BoostingStrategy;
import org.apache.spark.mllib.tree.model.GradientBoostedTreesModel;
import scala.Tuple2;

import java.util.HashMap;
import java.util.Map;


public class GBTTitanicClassification {
    public static void main(String[] args) {
        //Hide the Log
        LogManager.getLogger("org").setLevel(org.apache.log4j.Level.OFF);

        SparkConf sparkConf = new SparkConf().setMaster("local[*]").setAppName("TitanicClassifier");
        JavaSparkContext javaSparkContext = new JavaSparkContext(sparkConf);

        //Load csv file into RDD; cut the head off
        JavaRDD<LabeledPoint> trainData = loadDataAndFormat(javaSparkContext, "titanic_train.csv");
        JavaRDD<LabeledPoint> testData = loadDataAndFormat(javaSparkContext, "titanic_test.csv");
        //System.out.println(trainData.count());
        //System.out.println(testData.count());

        // Train a GradientBoostedTrees model.
        // The defaultParams for Classification use LogLoss by default.
        BoostingStrategy boostingStrategy = BoostingStrategy.defaultParams("Classification");
        boostingStrategy.setNumIterations(100);
        boostingStrategy.getTreeStrategy().setMaxDepth(5);
        // Empty categoricalFeaturesInfo indicates all features are continuous
        Map<Integer, Integer> categoricalFeaturesInfo = new HashMap<>();
        boostingStrategy.treeStrategy().setCategoricalFeaturesInfo(categoricalFeaturesInfo);
        final GradientBoostedTreesModel model =
                GradientBoostedTrees.train(trainData, boostingStrategy);
        System.out.println("loss function: " + boostingStrategy.getLoss());
        System.out.println(model.numTrees() + "Tree number");
        // Evaluate model on test instances and compute test error
        JavaPairRDD<Object, Object> predictionAndLabel =
                testData.mapToPair(new PairFunction<LabeledPoint, Object, Object>() {
                    @Override
                    public Tuple2<Object, Object> call(LabeledPoint labeledPoint) throws Exception {
                        return new Tuple2<>(model.predict(labeledPoint.features()), labeledPoint.label());
                    }
                });
        Double testErr =
                1.0 * predictionAndLabel.filter(new Function<Tuple2<Object, Object>, Boolean>() {
                    @Override
                    public Boolean call(Tuple2<Object, Object> pl) {
                        return !pl._1().equals(pl._2());
                    }
                }).count() / testData.count();

        //Test result evaluation
        //output the accuracy of the prediction
        System.out.println("\n============ Evaluation ===========");
        System.out.println("Test Error: " + testErr);
        System.out.println("Accuracy: " + (1 - testErr));

        //confusion matrix

        //count the true positive prediction
        long truePositive = predictionAndLabel.filter(new Function<Tuple2<Object, Object>, Boolean>() {
            @Override
            public Boolean call(Tuple2<Object, Object> pl) throws Exception {
                return pl._1().equals(pl._2()) && pl._1().equals(1d);
            }
        }).count();
        //count the true negative prediction
        long trueNegative = predictionAndLabel.filter(new Function<Tuple2<Object, Object>, Boolean>() {
            @Override
            public Boolean call(Tuple2<Object, Object> pl) throws Exception {
                return !pl._1().equals(pl._2()) && pl._1().equals(1d);
            }
        }).count();
        //count the false positive prediction
        long falsePositive = predictionAndLabel.filter(new Function<Tuple2<Object, Object>, Boolean>() {
            @Override
            public Boolean call(Tuple2<Object, Object> pl) throws Exception {
                return pl._1().equals(pl._2()) && pl._1().equals(0d);
            }
        }).count();
        //count the false negative prediction
        long falseNegative = predictionAndLabel.filter(new Function<Tuple2<Object, Object>, Boolean>() {
            @Override
            public Boolean call(Tuple2<Object, Object> pl) throws Exception {
                return !pl._1().equals(pl._2()) && pl._1().equals(0d);
            }
        }).count();

        // Confusion matrix
        System.out.println("===================================");
        System.out.println("Confusion matrix: " );
        System.out.println("\t\tReference\nPrediction\t0\t1" );
        System.out.println("\t0\t"+falsePositive+"\t" + falseNegative);
        System.out.println("\t1\t"+trueNegative+"\t" + truePositive);
        //https://spark.apache.org/docs/2.2.0/mllib-evaluation-metrics.html
        /*
            MulticlassMetrics McM = new MulticlassMetrics(predictionAndLabel.rdd());

            Matrix confusion = McM.confusionMatrix();
            System.out.println("Confusion matrix: \n" + confusion);
        */
        System.out.println("===================================");
        BinaryClassificationMetrics bcm = new BinaryClassificationMetrics(predictionAndLabel.rdd());
        System.out.println("AUC: ");
        // AUROC
        System.out.println("Area under ROC = " + bcm.areaUnderROC() + "\n\n");

        javaSparkContext.close();
    }

    //Load the data into spark RDD and omit the header
    private static JavaRDD<LabeledPoint> loadDataAndFormat(JavaSparkContext javaSparkContext, String path){
        JavaRDD<String> data =  javaSparkContext.textFile(path);
        //Omit header
        final String header = data.first();
        JavaRDD<String> dataWithoutHeader = data.filter(new Function<String, Boolean>() {
            @Override
            public Boolean call(String s) throws Exception {
                return !s.equalsIgnoreCase(header);
            }
        });
        //convert it into labelpointer
        //https://stackoverflow.com/questions/39530775/apache-spark-mllib-getting-labeledpoint-from-data-java
        JavaRDD<LabeledPoint> formatedData = dataWithoutHeader.map(new Function<String, LabeledPoint>() {
            @Override
            public LabeledPoint call(String line) throws Exception {
                String[] elements = line.split(",");
                return new LabeledPoint(Double.valueOf(elements[1]),
                                        Vectors.dense(Double.valueOf(elements[2]),
                                                      Double.valueOf(elements[3]),
                                                      Double.valueOf(elements[4]),
                                                      Double.valueOf(elements[5])));
            }
        });
        return formatedData;
    }
}
