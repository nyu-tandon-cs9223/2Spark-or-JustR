import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics;
import org.apache.spark.mllib.linalg.Vectors;
import scala.Tuple2;
import java.util.Map;
import java.util.HashMap;
import org.apache.spark.api.java.*;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.tree.RandomForest;
import org.apache.spark.mllib.tree.model.RandomForestModel;
import org.apache.spark.SparkConf;
import org.apache.log4j.Logger;
import org.apache.log4j.Level;
public class JavaRandomForestClassificationMetricsExample {
    public static void main(String[] args) {
        Logger.getLogger("org").setLevel(Level.OFF);
        Logger.getLogger("akka").setLevel(Level.OFF);
        // $example on$
        SparkConf conf = new SparkConf().set("spark.executor.memory", "2g")
                .set("spark.driver.memory", "8g").setMaster("local[*]").setAppName("randomForest");
        JavaSparkContext sc = new JavaSparkContext(conf);
        // Load and the data file.
        String trPath = "trdata_nohead.csv";
        String tstPath = "tstdata_nohead.csv";
        JavaRDD<String> trdata = sc.textFile(trPath);
        JavaRDD<String> tstdata = sc.textFile(tstPath);
        JavaRDD<LabeledPoint> training = trdata.map(new Function<String, LabeledPoint>() {
            @Override
            public LabeledPoint call(String v1) throws Exception {
                String[] parts = v1.split(",(?=([^\"]*\"[^\"]*\")*[^\"]*$)");
                Double sex = parts[2].equals("male") ? 1.0 : 0.0;
                return new LabeledPoint(Double.parseDouble(parts[0]),
                        Vectors.dense(Double.parseDouble(parts[1]),
                                sex,
                                Double.parseDouble(parts[3]),
                                Double.parseDouble(parts[4])
                        ));
            }
        });
        JavaRDD<LabeledPoint> test = tstdata.map(new Function<String, LabeledPoint>() {
            @Override
            public LabeledPoint call(String v1) throws Exception {
                String[] parts = v1.split(",(?=([^\"]*\"[^\"]*\")*[^\"]*$)");
                Double sex = parts[2].equals("male")? 1.0 : 0.0;
                return new LabeledPoint(Double.parseDouble(parts[0]),
                        Vectors.dense(Double.parseDouble(parts[1]),
                                sex,
                                Double.parseDouble(parts[3]),
                                Double.parseDouble(parts[4])
                        ));
            }
        });
        // Train a RandomForest model.
        // Empty features 2 and 3 are continuous.
        Integer numClasses = 2;
        Map<Integer, Integer> categoricalFeaturesInfo = new HashMap<Integer, Integer>(){{
            put(0,4);
            put(1,3);
        }};
        Integer numTrees = 200;
        String featureSubsetStrategy = "auto";
        String impurity = "gini";
        Integer maxDepth = 30;
        Integer maxBins = 10;
        Integer seed = 43;
        final RandomForestModel model = RandomForest.trainClassifier(training, numClasses,
                categoricalFeaturesInfo, numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins,
                seed);
        //Test the model
        JavaPairRDD<Object, Object> predictionsAndLabel = test.mapToPair(
                p -> new Tuple2<Object, Object>(model.predict(p.features()), p.label()));
        MulticlassMetrics metrics = new MulticlassMetrics(predictionsAndLabel.rdd());
        System.out.println("Confusion Matrix:\n"+metrics.confusionMatrix());
        System.out.println("Accuracy:"+metrics.accuracy());
        BinaryClassificationMetrics metrics_ = new BinaryClassificationMetrics(predictionsAndLabel.rdd());
        System.out.println("AUC:"+metrics_.areaUnderROC());
        sc.stop();
    }
}

