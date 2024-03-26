# -*- coding: utf-8 -*-
!apt-get install openjdk-8-jdk-headless -qq > /dev/null
!wget -q http://archive.apache.org/dist/spark/spark-3.1.1/spark-3.1.1-bin-hadoop3.2.tgz
!tar xf spark-3.1.1-bin-hadoop3.2.tgz
!pip install -q findspark
!pip install mlflow
!pip install pyngrok

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor  # Import GBTRegressor for XGBoost-like functionality
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.feature import PCA
from pyspark.sql.functions import col
import pyspark.sql.functions as F
import mlflow
import subprocess
from pyngrok import ngrok, conf
import getpass
import os

os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
os.environ["SPARK_HOME"] = "spark-3.1.1-bin-hadoop3.2"

import findspark
findspark.init()

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
subprocess.Popen(["mlflow", "ui", "--backend-store-uri", MLFLOW_TRACKING_URI])

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("XGB_experiment1")  # Update experiment name

spark = SparkSession.builder.appName("xgb_model").config("spark.executor.memory", "4g").config("spark.driver.memory", "4g").config("spark.executor.cores", "2").config("spark.default.parallelism", "4").getOrCreate()

# Read the Parquet file into a DataFrame
df_sample = spark.read.parquet("rollup_data")
embedding_length = 384

# Start MLflow run
with mlflow.start_run(run_name="XGBoostRegressor_PCA"):  # Update run name

    # Check if 'Impact' is float and convert it to double
    df_sample = df_sample.withColumn('Impact', col('Impact').cast('double'))  # Fix the variable name

    # Select embedding variables and 'Impact' for modeling
    feature_columns = [item for item in df_sample.columns if item not in ['Index', 'publisher', 'Impact']]
    df_model = df_sample.select(feature_columns + ['Impact'])

    # Use PCA for dimensionality reduction to 50 components
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    pca = PCA(k=50, inputCol="features", outputCol="pca_features")

    # Train-test split
    train_df, test_df = df_model.randomSplit([0.7, 0.3], seed=42)

    # Create a GBT Regressor model (similar to XGBoost)
    xgb = GBTRegressor(labelCol="Impact", featuresCol="pca_features", seed=42)  # Use GBTRegressor for XGBoost-like functionality

    # Create a pipeline
    pipeline = Pipeline(stages=[assembler, pca, xgb])

    # Define a parameter grid for tuning
    paramGrid = ParamGridBuilder().addGrid(xgb.maxDepth, [5, 10, 15]).addGrid(xgb.maxIter, [10, 20, 30]).build()

    # Create a cross-validator
    crossval = CrossValidator(estimator=pipeline,
                              estimatorParamMaps=paramGrid,
                              evaluator=RegressionEvaluator(labelCol="Impact"),
                              numFolds=3)

    # Fit the model
    cv_model = crossval.fit(train_df)

    # Log parameters
    mlflow.log_params({
        "maxDepth": cv_model.bestModel.stages[2].getMaxDepth(),
        "maxIter": cv_model.bestModel.stages[2].getMaxIter()
    })

    # Log metrics
    train_predictions = cv_model.transform(train_df)
    test_predictions = cv_model.transform(test_df)

    # Calculate MAPE for training set
    train_mape = train_predictions.withColumn('mape', F.abs((col('Impact') - col('prediction')) / col('Impact')) * 100)
    training_mape = train_mape.select(F.mean('mape')).collect()[0][0]
    mlflow.log_metric("training_mape", training_mape)

    # Calculate MAPE for test set
    test_mape = test_predictions.withColumn('mape', F.abs((col('Impact') - col('prediction')) / col('Impact')) * 100)
    test_mape = test_mape.select(F.mean('mape')).collect()[0][0]
    mlflow.log_metric("test_mape", test_mape)

    # Log the model
    mlflow.spark.log_model(cv_model.bestModel, "model")

    # Show MAPE on training set
    print(f"MAPE on training set: {training_mape:.2f}%")

    # Show MAPE on test set
    print(f"MAPE on test set: {test_mape:.2f}%")
