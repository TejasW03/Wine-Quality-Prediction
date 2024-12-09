package com.example;

import org.apache.spark.sql.*;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;

import java.io.IOException;

public class App {
    public static void main(String[] args) {
        // Set up the Spark session
        SparkSession sparkSession = SparkSession.builder()
                .appName("WineQualityPredictorApp")
                .getOrCreate();

        // Load the training dataset
        Dataset<Row> trainingDataset = sparkSession.read()
                .option("header", "true")  // Ensure the first row is treated as column names
                .option("inferSchema", "true") // Auto-detect data types
                .option("delimiter", ";")  // Use ';' as the column separator
                .option("quote", "\"")     // Handle quoted headers
                .csv("file:///home/ubuntu/TrainingDataset.csv");

        // Load the validation dataset
        Dataset<Row> validationDataset = sparkSession.read()
                .option("header", "true")
                .option("inferSchema", "true")
                .option("delimiter", ";")
                .option("quote", "\"")
                .csv("file:///home/ubuntu/ValidationDataset.csv");

        // Debug: Display dataset schemas
        System.out.println("Schema of Training Dataset:");
        trainingDataset.printSchema();
        System.out.println("Schema of Validation Dataset:");
        validationDataset.printSchema();

        // Clean column names by stripping quotes
        Dataset<Row> cleanedTrainingData = cleanColumnNames(trainingDataset);
        Dataset<Row> cleanedValidationData = cleanColumnNames(validationDataset);

        // Create a vector assembler for feature aggregation
        VectorAssembler featureAssembler = new VectorAssembler()
                .setInputCols(new String[]{"fixed acidity", "volatile acidity", "citric acid",
                        "residual sugar", "chlorides", "free sulfur dioxide",
                        "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"})
                .setOutputCol("features");

        Dataset<Row> processedTrainingData = featureAssembler.transform(cleanedTrainingData);

        // Initialize and train the logistic regression model
        LogisticRegression logisticRegression = new LogisticRegression()
                .setLabelCol("quality")
                .setFeaturesCol("features");
        LogisticRegressionModel trainedModel = logisticRegression.fit(processedTrainingData);

        // Prepare validation data and generate predictions
        Dataset<Row> processedValidationData = featureAssembler.transform(cleanedValidationData);
        Dataset<Row> predictionResults = trainedModel.transform(processedValidationData);

        // Compute evaluation metrics using F1 score
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("quality")
                .setPredictionCol("prediction")
                .setMetricName("f1");

        double f1Metric = evaluator.evaluate(predictionResults);
        System.out.println("F1 Score of the Model: " + f1Metric);

        // Persist the trained model
        try {
            trainedModel.save("file:///home/ubuntu/models/wine_quality_model");
        } catch (IOException exception) {
            System.err.println("Model saving failed: " + exception.getMessage());
            exception.printStackTrace();
        }

        // Close the Spark session
        sparkSession.stop();
    }

    /**
     * Utility method to sanitize column names by removing extraneous quotes.
     */
    private static Dataset<Row> cleanColumnNames(Dataset<Row> dataset) {
        for (String columnName : dataset.columns()) {
            String sanitizedColumnName = columnName.replace("\"", "").trim();
            dataset = dataset.withColumnRenamed(columnName, sanitizedColumnName);
        }
        return dataset;
    }
}
