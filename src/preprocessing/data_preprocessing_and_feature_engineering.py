# -*- coding: utf-8 -*-
!apt-get install openjdk-8-jdk-headless -qq > /dev/null
!wget -q http://archive.apache.org/dist/spark/spark-3.1.1/spark-3.1.1-bin-hadoop3.2.tgz
!tar xf spark-3.1.1-bin-hadoop3.2.tgz
!pip install -q findspark
!pip install mlflow
!pip install pyngrok
!pip install sentence_transformers

import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
os.environ["SPARK_HOME"] = "spark-3.1.1-bin-hadoop3.2"

import findspark
findspark.init()

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, expr, coalesce, current_date, year, udf
from pyspark.ml.linalg import Vectors, VectorUDT
from sentence_transformers import SentenceTransformer
from pyspark.sql.types import ArrayType, StringType
from textblob import TextBlob
from pyspark.sql.functions import size, split, length, array_contains

# Create a Spark session
spark = SparkSession.builder.appName("BookEmbeddings1").config("spark.executor.memory", "4g").config("spark.driver.memory", "4g").config("spark.executor.cores", "2").config("spark.default.parallelism", "4").getOrCreate()

def embed_sentences(text):
    embeddings = embeddings_model.encode([text])
    return embeddings.tolist()[0]

# Register the UDF
spark.udf.register("embed_sentences", embed_sentences, "array<double>")

# Define a UDF to convert string representation of list to actual list
@udf(ArrayType(StringType()))
def convert_to_list(authors_str):
    return (authors_str[1:-1].replace("'", "").split(',')) if authors_str is not None else []

# Define a UDF for sentiment analysis using TextBlob
def analyze_sentiment(title):
    blob = TextBlob(title)
    return blob.sentiment.polarity

sentiment_udf = udf(analyze_sentiment, StringType())

embedding_length= 384

# Read data into DataFrame (need to differentiate between , and ",")
csv_path = "books_task.csv"
df = spark.read.option("header", "true").option("quote", "\"").option("escape", "\"").csv(csv_path)


#### Data preprocessing ####
#==========================#

# Rename the '_c0' column to 'Index'
df = df.withColumnRenamed("_c0", "Index")

# Count missing values in each column
missing_counts = [df.where(col(c).isNull()).count() for c in df.columns]

# Create a dictionary to map column names to missing value counts
missing_dict = dict(zip(df.columns, missing_counts))

# Display the missing value counts for each column
for column, missing_count in missing_dict.items():
    print(f"Column '{column}': {missing_count} missing values")

# Convert 'publishedDate' to date format
df = df.withColumn("publishedDate", col("publishedDate").cast("date"))

# Replace missing values with a default date
default_date = "1900-01-01"
df = df.withColumn("publishedDate", when(col("publishedDate").isNull(), default_date).otherwise(col("publishedDate")))

# Apply the UDF to the 'authors' column to get authors_list
df = df.withColumn("authors_list", convert_to_list(col("authors")))

# Apply the UDF to the 'categories' column to get categories_list
df = df.withColumn("categories_list", convert_to_list(col("categories")))

# Extract all unique authors
all_authors = df.select("authors_list").rdd.flatMap(lambda x: x[0]).distinct().collect()
print(f"Total no. of authors:  '{len(all_authors)}'")
# ~120k - Too many authors so leaving out

# Print unique publishers
unique_publishers = df.select("publisher").distinct().rdd.flatMap(lambda x: x).collect()
print(f"Total no. of authors:  '{len(unique_publishers)}'")

# Convert Impact from string to double
df = df.withColumn("Impact", col("Impact").cast("double"))


#### Feature Engineering ####
#===========================#

# get publishedYear to calculate age of book
df = df.withColumn("publishedYear", year("publishedDate"))
df = df.withColumn("age", year(current_date()) - year("publishedDate"))

# coalesce - title and description
df = df.withColumn("text", coalesce(col("Title"), col("description")))

# load embeddings_model
embeddings_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Apply the UDF to create a new text column with embeddings
df = df.withColumn("text_embeddings", expr("embed_sentences(text)"))

# Create separate columns for each dimension of the text_embeddings
for i in range(embedding_length):
        col_name = f"text_embeddings{i + 1}"
        df = df.withColumn(col_name, df["text_embeddings"][i])

# Multi Hot encoding for all_categories
all_categories = df.select("categories_list").rdd.flatMap(lambda x: x[0]).distinct().collect()
for i in range(len(all_categories)):
    # Use the 'when' function to create a binary column
    df = df.withColumn(f"category_{i}", when(array_contains(col("categories_list"), all_categories[i]), 1).otherwise(0))

# title features - title_word_count, title_avg_word_length, char_count, contains_digits, contains_uppercase, contains_punctuation,
df = df.withColumn("title_word_count", size(split(col("Title"), " ")))
df = df.withColumn("title_avg_word_length", length(col("Title")) / col("title_word_count"))
df = df.withColumn("char_count", length(col("Title")))
df = df.withColumn("contains_digits", when(col("Title").rlike("\\d"), 1).otherwise(0))
df = df.withColumn("contains_uppercase", when(col("Title").rlike("[A-Z]"), 1).otherwise(0))
df = df.withColumn("contains_punctuation", when(col("Title").rlike("[^\w\s]"), 1).otherwise(0))

# Apply the sentiment_score UDF to the 'Title' column to get sentiment_score
df = df.withColumn("sentiment_score", sentiment_udf(col("Title")))
df = df.withColumn("sentiment_score", col("sentiment_score").cast("double"))

columns_to_drop = ['description','categories_list', 'categories','publishedDate', 'publishedYear','authors_list', 'authors','Title','text','text_embeddings']
df = df.drop(*columns_to_drop)

df.write.parquet("rollup_data")

