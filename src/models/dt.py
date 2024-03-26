# -*- coding: utf-8 -*-
!apt-get install openjdk-8-jdk-headless -qq > /dev/null
!wget -q http://archive.apache.org/dist/spark/spark-3.1.1/spark-3.1.1-bin-hadoop3.2.tgz
!tar xf spark-3.1.1-bin-hadoop3.2.tgz
!pip install -q findspark
!pip install mlflow
!pip install pyngrok

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
import pyspark.sql.functions as F
import mlflow
import subprocess
import os

os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
os.environ["SPARK_HOME"] = "spark-3.1.1-bin-hadoop3.2"

import findspark
findspark.init()

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
subprocess.Popen(["mlflow", "ui", "--backend-store-uri", MLFLOW_TRACKING_URI])

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("DecisionTree_experiment1")

spark = SparkSession.builder.appName("dt_model").config("spark.executor.memory", "4g").config("spark.driver.memory", "4g").config("spark.executor.cores", "2").config("spark.default.parallelism", "4").getOrCreate()

# Read the Parquet file into a DataFrame
df_sample = spark.read.parquet("rollup_data")

# Start MLflow run
with mlflow.start_run(run_name="DecisionTree_NoPCA"):

    # Check if 'Impact' is float and convert it to double
    df_sample = df_sample.withColumn('Impact', col('Impact').cast('double'))

    # Select embedding variables and 'Impact' for modeling
    feature_columns = [item for item in df_sample.columns if item not in ['Index','publisher','Impact']]
    df_model = df_sample.select(feature_columns + ['Impact'])

    # Train-test split
    train_df, test_df = df_model.randomSplit([0.7, 0.3], seed=42)

    # Create a Decision Tree regression model
    dt = DecisionTreeRegressor(labelCol="Impact", featuresCol="features")

    # Create a Vector Assembler
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")

    # Create a pipeline
    pipeline = Pipeline(stages=[assembler, dt])

    # Define a parameter grid for tuning (if needed)
    paramGrid = ParamGridBuilder().addGrid(dt.maxDepth, [5, 10, 15]).build()

    # Create a cross-validator (if needed)
    crossval = CrossValidator(estimator=pipeline,
                              estimatorParamMaps=paramGrid,
                              evaluator=RegressionEvaluator(labelCol="Impact"),
                              numFolds=3)

    # Fit the model
    model = pipeline.fit(train_df)

    # Log parameters
    mlflow.log_params({
        "maxDepth": model.stages[1].getMaxDepth()
    })

    # Log metrics
    train_predictions = model.transform(train_df)
    test_predictions = model.transform(test_df)

    # Calculate MAPE for training set
    train_mape = train_predictions.withColumn('mape', F.abs((col('Impact') - col('prediction')) / col('Impact')) * 100)
    training_mape = train_mape.select(F.mean('mape')).collect()[0][0]
    mlflow.log_metric("training_mape", training_mape)

    # Calculate MAPE for test set
    test_mape = test_predictions.withColumn('mape', F.abs((col('Impact') - col('prediction')) / col('Impact')) * 100)
    test_mape = test_mape.select(F.mean('mape')).collect()[0][0]
    mlflow.log_metric("test_mape", test_mape)

    # Log the model
    mlflow.spark.log_model(model, "model")

    # Show MAPE on training set
    print(f"MAPE on training set: {training_mape:.2f}%")

    # Show MAPE on test set
    print(f"MAPE on test set: {test_mape:.2f}%")
