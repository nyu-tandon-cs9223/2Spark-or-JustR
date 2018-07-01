import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.apache.log4j.LogManager;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.classification.SVMModel;
import org.apache.spark.mllib.classification.SVMWithSGD;
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.SingularValueDecomposition;
import org.apache.spark.mllib.regression.LabeledPoint;
import scala.Tuple2;

import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.distributed.RowMatrix;

/**
 * Created by Jiwei Yu on 11/8/17.
 */
public class Svm {
    public static void main(String[] args) {
    	LogManager.getLogger("org").setLevel(org.apache.log4j.Level.OFF);
    	
        SparkConf sparkConf = new SparkConf().setAppName("RandomForestTitanic").setMaster("local[1]");
        JavaSparkContext jsc = new JavaSparkContext(sparkConf);
        
        //get training data
        JavaRDD<String> trData = jsc.textFile("trdata.csv").filter(line ->
                !"".equals(line.split(",(?=([^\"]*\"[^\"]*\")*[^\"]*$)")[5]));
        
        JavaRDD<LabeledPoint> training = trData.map(line -> {
            String[] params = line.split(",(?=([^\"]*\"[^\"]*\")*[^\"]*$)");
            double label = Double.valueOf(params[1]);
            double[] vector = new double[4];
            vector[0] = Double.valueOf(params[2]);
            vector[1] = "male".equals(params[3]) ? 1d : 0d;
            vector[2] = Double.valueOf(params[4]);
            vector[3] = Double.valueOf(params[5]);
            return new LabeledPoint(label, new DenseVector(vector));
        });
        
      //get test data
        JavaRDD<String> tstData = jsc.textFile("tstdata.csv").filter(line ->
                !"".equals(line.split(",(?=([^\"]*\"[^\"]*\")*[^\"]*$)")[5]));
        
        JavaRDD<LabeledPoint> test = tstData.map(line -> {
            String[] params = line.split(",(?=([^\"]*\"[^\"]*\")*[^\"]*$)");
            double label = Double.valueOf(params[1]);
            double[] vector = new double[4];
            vector[0] = Double.valueOf(params[2]);
            vector[1] = "male".equals(params[3]) ? 1d : 0d;
            vector[2] = Double.valueOf(params[4]);
            vector[3] = Double.valueOf(params[5]);
            return new LabeledPoint(label, new DenseVector(vector));
        });
        

        // Split initial RDD into two... [60% training data, 40% testing data].
        //JavaRDD<LabeledPoint> training = data.sample(false, 0.6, 11L);
        //training.cache();
        //JavaRDD<LabeledPoint> test = data.subtract(training);

        // Run training algorithm to build the model.
        //when set numIterations to 170, the AUR is 0.71
        int numIterations = 170;
        SVMModel model = SVMWithSGD.train(training.rdd(), numIterations);

        // Clear the default threshold.
        model.clearThreshold();

        // Compute raw scores on the test set.
        JavaRDD<Tuple2<Object, Object>> scoreAndLabels = test.map(p ->
                new Tuple2<>(model.predict(p.features()), p.label()));
        
        double min = Double.MAX_VALUE;
        double max = Double.MIN_VALUE;
        
        final ArrayList<Double> predictTemp = new ArrayList<>();
        List<Tuple2<Object, Object>> listOfRDD = scoreAndLabels.collect();
        
        listOfRDD.forEach(a -> predictTemp.add((double)a._1));
        
        for (Object p : predictTemp) {
        	if ((double) p < min) {
        		min = (double) p;
         	}
        	
        	if ((double) p > max) {
        		max = (double) p;
        	}
        }
        final double finalMin = min;
        final double finalMax = max;
        System.out.println(min + " "+ max);
        
        double testErr = scoreAndLabels.filter(pl -> {
	        	double norm = ((double) pl._1 - finalMin) / (finalMax - finalMin);
//	        	System.out.println(norm);
        		if ((norm > 0.5 ? 1.0d : 0.0d) == (double) pl._2()) {
	        		return false;
	        	} else {
	        		return true;
	        	}
        	}).count() / (double) test.count();
        
        System.out.println("testErr = " + testErr);
        
        // Get evaluation metrics.
        BinaryClassificationMetrics metrics =
                new BinaryClassificationMetrics(JavaRDD.toRDD(scoreAndLabels));
        double auROC = metrics.areaUnderROC();
        

// 	MulticlassMetrics not working  
//	MulticlassMetrics metricsm = new MulticlassMetrics(JavaRDD.toRDD(scoreAndLabels));
//	System.out.println("Confusion Matrix: \n" + metricsm.confusionMatrix());      


        System.out.println("Area under ROC = " + auROC);
        
        jsc.stop();
    }
}
