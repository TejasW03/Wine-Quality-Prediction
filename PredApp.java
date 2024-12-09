
package com.example;

import org.apache.spark.sql.*;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;

import java.io.PrintWriter;

public class App {
    public static void main(String[] args) {
        // Define file paths
        String trainedModelPath = "file:/home/ubuntu/models/wine_quality_model-tejas";
        String testDatasetPath = "file:/home/ubuntu/datasets/TestDataset.csv";

        // Initialize Spark session
        SparkSession sparkSession = SparkSession.builder()
                .appName("WineQualityEvaluationApp")
                .master("local[*]")
                .getOrCreate();

        // Load the test dataset
        Dataset<Row> rawTestDataset = sparkSession.read()
                .option("header", "true")
                .option("inferSchema", "true")
                .option("delimiter", ";")
                .csv(testDatasetPath);

        // Clean column headers for consistent naming
        Dataset<Row> cleanedTestDataset = standardizeColumnNames(rawTestDataset);

        // Prepare feature vector
        VectorAssembler featureAssembler = new VectorAssembler()
                .setInputCols(new String[]{
                        "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
                        "chlorides", "free_sulfur_dioxide",
                        "total_sulfur_dioxide", "density",
                        "pH", "sulphates", "alcohol"})
                .setOutputCol("features");

        Dataset<Row> testFeaturesDataset = featureAssembler.transform(cleanedTestDataset);

        // Load the pre-trained logistic regression model
        LogisticRegressionModel logisticModel = LogisticRegressionModel.load(trainedModelPath);

        // Generate predictions
        Dataset<Row> predictionResults = logisticModel.transform(testFeaturesDataset);

        // Evaluate predictions using the F1 score
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("quality")
                .setPredictionCol("prediction")
                .setMetricName("f1");

        double computedF1Score = evaluator.evaluate(predictionResults);

        // Display the F1 Score in the console
        System.out.println("Computed F1 Score: " + computedF1Score);

        // Save predictions to a CSV file
        predictionResults.select("quality", "prediction")
                .write()
                .option("header", "true")
                .csv("predictions_output");

        // Save the F1 score to a text file
        try (PrintWriter fileWriter = new PrintWriter("f1_score_predict.txt", "UTF-8")) {
            fileWriter.println("Computed F1 Score: " + computedF1Score);
        } catch (Exception exception) {
            exception.printStackTrace();
        }

        // Close the Spark session
        sparkSession.stop();
    }

    /**
     * Standardize column names by removing quotes, trimming whitespace, and replacing spaces with underscores.
     */
    private static Dataset<Row> standardizeColumnNames(Dataset<Row> dataset) {
        StructType schema = dataset.schema();
        String[] adjustedColumnNames = schema.fieldNames();

        for (int index = 0; index < adjustedColumnNames.length; index++) {
            adjustedColumnNames[index] = adjustedColumnNames[index]
                    .replaceAll("\"", "")
                    .trim()
                    .replace(" ", "_");
        }

        for (int index = 0; index < adjustedColumnNames.length; index++) {
            dataset = dataset.withColumnRenamed(schema.fieldNames()[index], adjustedColumnNames[index]);
        }

        return dataset;
    }
}
