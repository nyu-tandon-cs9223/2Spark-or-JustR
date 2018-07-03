/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *

 */

import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS;
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics;
import scala.Tuple2;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.rdd.RDD;

public class RidgeRegression {
    public static void main(String[] args) {
        SparkConf conf = new SparkConf().setAppName("Java Ridge Michael");
        SparkContext sc = new SparkContext(conf);

        //80/20 split for train and test
        String pathTrainData = "./train.csv";
        String pathTestData = "./test.csv";
        int minPartition = 1;
        RDD<String> input = sc.textFile(pathTrainData, minPartition);
        JavaRDD<String> trainData = input.toJavaRDD();
        RDD<String> input2 = sc.textFile(pathTestData, minPartition);
        JavaRDD<String> testData = input2.toJavaRDD();

        //convert input to RDD label points
        JavaRDD<LabeledPoint> training = trainData
                .map(new Function<String, LabeledPoint>() {
                    public LabeledPoint call(String line) throws Exception {
                        String[] parts = line.split(",");
                        return new LabeledPoint(Double.parseDouble(parts[0]),
                                Vectors.dense(Double.parseDouble(parts[1]),
                                        Double.parseDouble(parts[2]),
                                        Double.parseDouble(parts[3]),
                                        Double.parseDouble(parts[4])));
                    }
                });

        JavaRDD<LabeledPoint> test = testData
                .map(new Function<String, LabeledPoint>() {
                    public LabeledPoint call(String line) throws Exception {
                        String[] parts = line.split(",");
                        return new LabeledPoint(Double.parseDouble(parts[0]),
                                Vectors.dense(Double.parseDouble(parts[1]),
                                        Double.parseDouble(parts[2]),
                                        Double.parseDouble(parts[3]),
                                        Double.parseDouble(parts[4])));
                    }
                });

        // Run training algorithm to build the model.
        LogisticRegressionModel model = new LogisticRegressionWithLBFGS()
                .run(training.rdd());

        // Clear the prediction threshold so the model will return probabilities
        model.clearThreshold();

        // Compute raw scores on the test set.
        JavaRDD<Tuple2<Object, Object>> predictionAndLabels = test.map(
                new Function<LabeledPoint, Tuple2<Object, Object>>() {
                    public Tuple2<Object, Object> call(LabeledPoint p) {
                        Double prediction = (double)Math.round(model.predict(p.features()));
                        return new Tuple2<Object, Object>(prediction, p.label());
                    }
                }
        );

        // Get evaluation metrics.
        BinaryClassificationMetrics metrics =
                new BinaryClassificationMetrics(predictionAndLabels.rdd());

        // Get evaluation metrics.
        MulticlassMetrics metrics2 = new MulticlassMetrics(predictionAndLabels.rdd());
        // Accuracy
        System.out.println("Accuracy = " + metrics2.accuracy());

        // Precision by threshold
        JavaRDD<Tuple2<Object, Object>> precision = metrics.precisionByThreshold().toJavaRDD();
        System.out.println("Precision by threshold: " + precision.collect());

        // Recall by threshold
        JavaRDD<?> recall = metrics.recallByThreshold().toJavaRDD();
        System.out.println("Recall by threshold: " + recall.collect());

        // F Score by threshold
        JavaRDD<?> f1Score = metrics.fMeasureByThreshold().toJavaRDD();
        System.out.println("F1 Score by threshold: " + f1Score.collect());

        JavaRDD<?> f2Score = metrics.fMeasureByThreshold(2.0).toJavaRDD();
        System.out.println("F2 Score by threshold: " + f2Score.collect());

        // Precision-recall curve
        JavaRDD<?> prc = metrics.pr().toJavaRDD();
        System.out.println("Precision-recall curve: " + prc.collect());

        // Thresholds
        JavaRDD<Double> thresholds = precision.map(t -> Double.parseDouble(t._1().toString()));

        // ROC Curve
        JavaRDD<?> roc = metrics.roc().toJavaRDD();
        System.out.println("ROC curve: " + roc.collect());

        // AUPRC
        System.out.println("Area under precision-recall curve = " + metrics.areaUnderPR());

        // AUROC
        System.out.println("Area under ROC = " + metrics.areaUnderROC());

        // Save and load model

        //model.save(sc, "target/tmp/RidgeLogisticRegressionModel");
        //LogisticRegressionModel.load(sc, "target/tmp/RidgeLogisticRegressionModel");

        sc.stop();
    }
}