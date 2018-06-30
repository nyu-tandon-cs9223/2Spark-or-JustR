import org.apache.hadoop.fs.FileSystem;
import org.apache.log4j.LogManager;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.Vectors;
import scala.Tuple2;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.DecisionTree;
import org.apache.spark.mllib.tree.model.DecisionTreeModel;
import org.apache.spark.mllib.util.MLUtils;
import java.util.HashMap;
import java.util.Map;
import org.apache.spark.SparkConf;
import sun.misc.FloatingDecimal;

/**
 ** Classification using Decision Trees in Apache Spark MLlib with Java Example
 */
public class DecisionTreeClassifierExample {


    public static final String COMMA_DELIMITER = ",(?=([^\"]*\"[^\"]*\")*[^\"]*$)";

    public static void main(String[] args) {

        LogManager.getLogger("org").setLevel(org.apache.log4j.Level.OFF);
        SparkConf sparkConf = new SparkConf().setMaster("local[*]").setAppName("Titanic Spark");

        JavaSparkContext javaSparkContext = new JavaSparkContext(sparkConf);

        JavaRDD<String> trainData =
            javaSparkContext.textFile("titanic_train.csv").filter(line ->
                !"".equals(line.split(COMMA_DELIMITER)[5]));

        JavaRDD<String> testingData =
            javaSparkContext.textFile("titanic_test.csv").filter(line ->
                !"".equals(line.split(COMMA_DELIMITER)[5]));

        JavaRDD<LabeledPoint> trainingData =
            trainData.map(
                line -> { String[] params = line.split(COMMA_DELIMITER);
                    double label = Double.valueOf(params[2]);
                    double[] vector = new double[4];
                    vector[0] = Double.valueOf(params[3]);
                    vector[1] = "male".equals(params[5]) ? 1d : 0d;
                    vector[2] = Double.valueOf(params[6]);
                    vector[3] = Double.valueOf(params[10]);
                    return new LabeledPoint(label, new DenseVector(vector)); });

        JavaRDD<LabeledPoint>
            testData = testingData.map(
            line -> { String[] params = line.split(COMMA_DELIMITER);
                double label = Double.valueOf(params[2]);
                double[] vector = new double[4];
                vector[0] = Double.valueOf(params[3]);
                vector[1] = "male".equals(params[5]) ? 1d : 0d;
                vector[2] = Double.valueOf(params[6]);
                vector[3] = Double.valueOf(params[10]);
                return new LabeledPoint(label, new DenseVector(vector)); });

        // Empty categoricalFeaturesInfo indicates all featuresare continuous.
        int numClasses = 2;
        Map<Integer, Integer> categoricalFeaturesInfo = new HashMap<>();
        String impurity = "gini";
        int maxDepth = 5;
        int maxBins = 32;

        // Train a Decision Tree model
        DecisionTreeModel model =
        DecisionTree.trainClassifier(trainingData, numClasses, categoricalFeaturesInfo, impurity, maxDepth, maxBins);

        // Predict for the test data using the model trained
        JavaPairRDD<Object,Object> predictionAndLabel = testData.mapToPair(p -> new Tuple2<>(model.predict(p.features()), p.label()));

        // calculate the accuracy
        double accuracy = predictionAndLabel.filter(pl -> pl._1().equals(pl._2())).count()
            / (double) testData.count();
        System.out.println("Accuracy is : "+accuracy);

        // multiclass
        // Get evaluation metrics.
        MulticlassMetrics metrics = new MulticlassMetrics(predictionAndLabel.rdd());


        // Overall statistics
        System.out.println("Accuracy = " + metrics.accuracy());

        // Confusion matrix
        Matrix confusion = metrics.confusionMatrix();

        System.out.println("Confusion matrix: \n" + confusion); // Stats by labels

        for(int i = 0; i < metrics.labels().length; i++) {
            System.out.format("Class %f precision = %f\n", metrics.labels()[i],metrics.precision( metrics.labels()[i]));
            System.out.format("Class %f recall = %f\n", metrics.labels()[i], metrics.recall(
                metrics.labels()[i])); System.out.format("Class %f F1 score = %f\n",
                metrics.labels()[i], metrics.fMeasure( metrics.labels()[i])); } //Weighted stats
        System.out.format("Weighted precision = %f\n", metrics.weightedPrecision());
        System.out.format("Weighted recall = %f\n", metrics.weightedRecall());
        System.out.format("Weighted F1 score = %f\n", metrics.weightedFMeasure());
        System.out.format("Weighted false positive rate = %f\n",
            metrics.weightedFalsePositiveRate()); // // Get evaluation metrics.

        BinaryClassificationMetrics metrics1 = new
            BinaryClassificationMetrics(predictionAndLabel.rdd()); // Precision by threshold

        JavaRDD<Tuple2<Object, Object>> precision =
            metrics1.precisionByThreshold().toJavaRDD(); // Thresholds

        JavaRDD<Double> thresholds = precision.map(t -> Double.parseDouble(t._1().toString()));
        //
        // ROC Curve
        JavaRDD<?> roc = metrics1.roc().toJavaRDD();
        System.out.println("ROC curve: " + roc.collect()); //
        //
        // AUPRC
        System.out.println("Area under precision-recall curve = " + metrics1.areaUnderPR());
        // AUROC
        System.out.println("Area under ROC = " + metrics1.areaUnderROC());
        // stop the spark context
        javaSparkContext.stop(); } }


