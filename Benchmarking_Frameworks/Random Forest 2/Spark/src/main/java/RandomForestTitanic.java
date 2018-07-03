import org.apache.log4j.LogManager;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.RandomForest;
import org.apache.spark.mllib.tree.model.RandomForestModel;
import scala.Tuple2;

import java.util.HashMap;
import java.util.Map;


/**
 * Created by chaoqunhuang on 11/8/17.
 */
public class RandomForestTitanic {
    // a regular expression which matches commas but not commas within double quotations
    public static final String COMMA_DELIMITER = ",(?=([^\"]*\"[^\"]*\")*[^\"]*$)";

    public static void main(String[] args) {
        LogManager.getLogger("org").setLevel(org.apache.log4j.Level.OFF);
        SparkConf sparkConf = new SparkConf().setAppName("RandomForestTitanic").setMaster("local[1]");

        JavaSparkContext jsc = new JavaSparkContext(sparkConf);

        JavaRDD<String> testData = jsc.textFile("test.csv");
        JavaRDD<String> trainData = jsc.textFile("train.csv");

        JavaRDD<LabeledPoint> trainPoints = trainData.filter(l -> !"Survived".equals(l.split(COMMA_DELIMITER)[0])).map(line -> {
            String[] params = line.split(COMMA_DELIMITER);
            double label = Double.valueOf(params[0]);
            double[] vector = new double[4];
            vector[0] = Double.valueOf(params[1]);
            vector[1] = Double.valueOf(params[2]);
            vector[2] = Double.valueOf(params[3]);
            vector[3] = Double.valueOf(params[4]);
            return new LabeledPoint(label, new DenseVector(vector));
        });

        JavaRDD<LabeledPoint> testPoints = testData.filter(l -> !"Survived".equals(l.split(COMMA_DELIMITER)[0])).map(line -> {
            String[] params = line.split(COMMA_DELIMITER);
            double label = Double.valueOf(params[0]);
            double[] vector = new double[4];
            vector[0] = Double.valueOf(params[1]);
            vector[1] = Double.valueOf(params[2]);
            vector[2] = Double.valueOf(params[3]);
            vector[3] = Double.valueOf(params[4]);
            return new LabeledPoint(label, new DenseVector(vector));
        });

        System.out.println("total train data:" + trainPoints.count());
        System.out.println("total test:" + testPoints.count());

        // Train a RandomForest model
        // Empty categoricalFeaturesInfo indicates all features are continuous
        Integer numClasses = 2;
        Map<Integer, Integer> categoricalFeaturesInfo = new HashMap<>();
        Integer numTrees = 5; // Use more in practice.
        String featureSubsetStrategy = "auto"; // Let the algorithm choose.
        String impurity = "gini";
        Integer maxDepth = 4;
        Integer maxBins = 32;
        Integer seed = 12345;

        RandomForestModel model = RandomForest.trainClassifier(trainPoints, numClasses,
                categoricalFeaturesInfo, numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins,
                seed);

        // Evaluate model on test instances and compute test error
        JavaPairRDD<Object, Object> predictionAndLabel =
                testPoints.mapToPair(p -> new Tuple2<>(model.predict(p.features()), p.label()));
        double testErr =
                predictionAndLabel.filter(pl -> !(pl._1()).equals(pl._2())).count() / (double) testData.count();
        System.out.println("Test Error: " + testErr);
        //System.out.println("Learned classification forest model:\n" + model.toDebugString());

        BinaryClassificationMetrics metrics =
                new BinaryClassificationMetrics(predictionAndLabel.rdd());


        long truePositive = predictionAndLabel.filter(pl -> (pl._1()).equals(pl._2()) && pl._1().equals(1d)).count();
        System.out.println("True Positive: " + truePositive);

        long trueNegative = predictionAndLabel.filter(pl -> (pl._1()).equals(pl._2()) && pl._1().equals(0d)).count();
        System.out.println("True Negative: " + trueNegative);

        long falsePositive = predictionAndLabel.filter(pl -> !(pl._1()).equals(pl._2()) && pl._2().equals(0d)).count();
        System.out.println("False Positive: " + falsePositive);

        long falseNegative = predictionAndLabel.filter(pl -> !(pl._1()).equals(pl._2()) && pl._2().equals(1d)).count();
        System.out.println("False Negative:" + falseNegative);

        System.out.println("\nConfusion Matrix:");
        System.out.println("n: " + testData.count() + "   0" + "   1");
        System.out.println("0: " + "   " + trueNegative + "   " + falseNegative);
        System.out.println("1: " + "   " + falsePositive + "   " + truePositive + "\n");

        // AUPRC
        System.out.println("Area under precision-recall curve = " + metrics.areaUnderPR());

        // AUROC
        System.out.println("Area under ROC = " + metrics.areaUnderROC());

        jsc.stop();
    }
}
