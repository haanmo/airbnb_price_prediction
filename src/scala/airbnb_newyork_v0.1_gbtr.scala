// Databricks notebook source
///////// Gradient-boosted tree regression ///////

import org.apache.spark.ml.feature.StandardScaler

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.corr
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.evaluation.RegressionEvaluator

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.regression.{GBTRegressionModel, GBTRegressor}


val filePath = "/FileStore/tables/pypbvn701490920254143/listings.csv"
val listing = spark.read.option("header","true").option("inferSchema","true").csv(filePath)

// COMMAND ----------

display(listing)

// COMMAND ----------

val listingClean = listing.na.drop()
val assembler = new VectorAssembler()
assembler.setInputCols(Array("latitude", "longitude", "minimum_nights", "number_of_reviews", "reviews_per_month", "calculated_host_listings_count", "availability_365", "price"))
assembler.setOutputCol("features")
val listingFeatures = assembler.transform(listingClean)


// COMMAND ----------

display(listingFeatures)

// COMMAND ----------

val scaler = new StandardScaler()
  .setInputCol("features")
  .setOutputCol("scaledFeatures")
  .setWithStd(true)
  .setWithMean(true)

// COMMAND ----------

// Compute summary statistics by fitting the StandardScaler.
val scalerModel = scaler.fit(listingFeatures)

// COMMAND ----------

// Normalize each feature to have unit standard deviation.
val scaledData = scalerModel.transform(listingFeatures)
display(scaledData)

// COMMAND ----------

//////// Random Forest with "not scaled" data
// Automatically identify categorical features, and index them.
// Set maxCategories so features with > 4 distinct values are treated as continuous.
val featureIndexer = new VectorIndexer()
  .setInputCol("features")
  .setOutputCol("indexedFeatures")
  .setMaxCategories(7)
  .fit(scaledData)

// COMMAND ----------

// Split the data into training and test sets (30% held out for testing).
val Array(trainingData, testData) = scaledData.randomSplit(Array(0.7, 0.3))

// COMMAND ----------

// Train a GBT model.
val gbt = new GBTRegressor()
  .setLabelCol("price")
  .setFeaturesCol("indexedFeatures")
  .setMaxIter(10)

// COMMAND ----------

// Chain indexer and GBT in a Pipeline.
val pipeline = new Pipeline()
  .setStages(Array(featureIndexer, gbt))


// COMMAND ----------

// Train model. This also runs the indexer.
val model = pipeline.fit(trainingData)

// COMMAND ----------

// Make predictions.
val predictions = model.transform(testData)

// COMMAND ----------

// Select example rows to display.
predictions.select("prediction", "price", "features").show(5)

// COMMAND ----------

// Select (prediction, true label) and compute test error.
val evaluator = new RegressionEvaluator()
  .setLabelCol("price")
  .setPredictionCol("prediction")
  .setMetricName("rmse")
val rmse = evaluator.evaluate(predictions)
println("Root Mean Squared Error (RMSE) on test data = " + rmse)


// COMMAND ----------

val gbtModel = model.stages(1).asInstanceOf[GBTRegressionModel]
println("Learned regression GBT model:\n" + gbtModel.toDebugString)

// COMMAND ----------

evaluator.setMetricName("mae")
val mae = evaluator.evaluate(predictions)
println("Mean Absolute Error = " + "%6.3f".format(mae))

// COMMAND ----------

//////// Random Forest with "scaled" data
// Automatically identify categorical features, and index them.
// Set maxCategories so features with > 4 distinct values are treated as continuous.
val scaledFeaturesIndexer = new VectorIndexer()
  .setInputCol("scaledFeatures")
  .setOutputCol("indexedScaledFeatures")
  .setMaxCategories(7)
  .fit(scaledData)

// COMMAND ----------

// Train a GBT model.
val gbtScaled = new GBTRegressor()
  .setLabelCol("price")
  .setFeaturesCol("indexedScaledFeatures")
  .setMaxIter(10)

// COMMAND ----------

// Chain indexer and forest in a Pipeline.
val pipeline = new Pipeline()
  .setStages(Array(scaledFeaturesIndexer, gbtScaled))

// COMMAND ----------



// COMMAND ----------

// Train model. This also runs the indexer.
//val modelScaled = pipeline.fit(trainingData)
val modelScaled = pipeline.fit(tr)

// COMMAND ----------

val te = testData.drop("features")
display(te)

// COMMAND ----------

// Make predictions.
//val predictionsScaled = modelScaled.transform(testData)
val predictionsScaled = modelScaled.transform(te)

// COMMAND ----------

// Select example rows to display.
predictionsScaled.select("prediction", "price", "scaledFeatures").show(5)

// COMMAND ----------

// Select (prediction, true label) and compute test error.
/*
val evaluator = new RegressionEvaluator()
  .setLabelCol("price")
  .setPredictionCol("prediction")
  .setMetricName("rmse")
*/
evaluator.setMetricName("rmse")
val rmse = evaluator.evaluate(predictionsScaled)
println("Root Mean Squared Error (RMSE) on test data = " + rmse)

// COMMAND ----------

val rfModelScaled = modelScaled.stages(1).asInstanceOf[GBTRegressionModel]
println("Learned regression forest model:\n" + rfModelScaled.toDebugString)

// COMMAND ----------

evaluator.setMetricName("mae")
val mae = evaluator.evaluate(predictionsScaled)
println("Mean Absolute Error = " + "%6.3f".format(mae))

// COMMAND ----------


