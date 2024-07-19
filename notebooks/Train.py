# Databricks notebook source
# MAGIC %md
# MAGIC ## INSTALL MLCORE SDK

# COMMAND ----------

# DBTITLE 1,Installing MLCore SDK
# MAGIC %pip install sparkmeasure
# MAGIC %pip install google-auth
# MAGIC %pip install google-cloud-storage
# MAGIC %pip install azure-storage-blob
# MAGIC %pip install azure-identity
# MAGIC %pip install protobuf==3.17.2

# COMMAND ----------

from sparkmeasure import StageMetrics, TaskMetrics
from pyspark.sql import functions as F

taskmetrics = TaskMetrics(spark)
stagemetrics = StageMetrics(spark)

taskmetrics.begin()
stagemetrics.begin()

# COMMAND ----------

# DBTITLE 1,Load the YAML config
import yaml
import ast
import pickle
from MLCORE_SDK import mlclient
from pyspark.sql import functions as F

try:
    solution_config = (dbutils.widgets.get("solution_config"))
    solution_config = ast.literal_eval(solution_config)
except Exception as e:
    print(e)
    with open('../data_config/SolutionConfig.yaml', 'r') as solution_config:
        solution_config = yaml.safe_load(solution_config)  

# COMMAND ----------

# MAGIC %md
# MAGIC ## PERFORM MODEL TRAINING 

# COMMAND ----------

# DBTITLE 1,Imports
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import time
from sklearn.metrics import *
import json

# COMMAND ----------

try :
    env = dbutils.widgets.get("env")
except :
    env = "dev"
print(f"Input environment : {env}")

# COMMAND ----------

# DBTITLE 1,Input from the user
# GENERAL PARAMETERS
tracking_env = solution_config["general_configs"]["tracking_env"]
try :
    sdk_session_id = dbutils.widgets.get("sdk_session_id")
except :
    sdk_session_id = solution_config["general_configs"]["sdk_session_id"][tracking_env]

if sdk_session_id.lower() == "none":
    sdk_session_id = solution_config["general_configs"]["sdk_session_id"][tracking_env]
 
tracking_url = solution_config["general_configs"].get("tracking_url", None)
tracking_url = f"https://{tracking_url}" if tracking_url else None

# JOB SPECIFIC PARAMETERS
input_table_configs = solution_config["train"]["datalake_configs"]["input_tables"]
output_table_configs = solution_config["train"]["datalake_configs"]['output_tables']
model_configs = solution_config["train"]["model_configs"]
storage_configs = solution_config["train"]["storage_configs"]
feature_columns = solution_config['train']["feature_columns"]
target_columns = solution_config['train']["target_columns"]
test_size = solution_config['train']["test_size"]

# COMMAND ----------

def get_name_space(table_config):
    data_objects = {}
    for table_name, config in table_config.items() : 
        catalog_name = config.get("catalog_name", None)
        schema = config.get("schema", None)
        table = config.get("table", None)

        if catalog_name and catalog_name.lower() != "none":
            table_path = f"{catalog_name}.{schema}.{table}"
        else :
            table_path = f"{schema}.{table}"

        data_objects[table_name] = table_path
    
    return data_objects

# COMMAND ----------

input_table_paths = get_name_space(input_table_configs)
output_table_paths = get_name_space(output_table_configs)

# COMMAND ----------

ft_data = spark.sql(f"SELECT * FROM {input_table_paths['input_1']}")
gt_data = spark.sql(f"SELECT * FROM {input_table_paths['input_2']}")

# COMMAND ----------

ft_data.display()

# COMMAND ----------

gt_data.display()

# COMMAND ----------

# DBTITLE 1,Check if any filters related to date or hyper parameter tuning are passed.
try : 
    date_filters = dbutils.widgets.get("date_filters")
    print(f"Input date filter : {date_filters}")
    date_filters = json.loads(date_filters)
except :
    date_filters = {}

try : 
    hyperparameters = dbutils.widgets.get("hyperparameters")
    print(f"Input hyper parameters : {hyperparameters}")
    hyperparameters = json.loads(hyperparameters)
except :
    hyperparameters = {}

print(f"Data filters used in model train : {date_filters}, hyper parameters : {hyperparameters}")


# COMMAND ----------

if date_filters and date_filters['feature_table_date_filters'] and date_filters['feature_table_date_filters'] != {} :   
    ft_start_date = date_filters.get('feature_table_date_filters', {}).get('start_date',None)
    ft_end_date = date_filters.get('feature_table_date_filters', {}).get('end_date',None)
    if ft_start_date not in ["","0",None] and ft_end_date not in  ["","0",None] : 
        print(f"Filtering the feature data")
        ft_data = ft_data.filter(F.col("timestamp") >= int(ft_start_date)).filter(F.col("timestamp") <= int(ft_end_date))

if date_filters and date_filters['ground_truth_table_date_filters'] and date_filters['ground_truth_table_date_filters'] != {} : 
    gt_start_date = date_filters.get('ground_truth_table_date_filters', {}).get('start_date',None)
    gt_end_date = date_filters.get('ground_truth_table_date_filters', {}).get('end_date',None)
    if gt_start_date not in ["","0",None] and gt_end_date not in ["","0",None] : 
        print(f"Filtering the ground truth data")
        gt_data = gt_data.filter(F.col("timestamp") >= int(gt_start_date)).filter(F.col("timestamp") <= int(gt_end_date))

# COMMAND ----------

ft_data.count(), gt_data.count()

# COMMAND ----------

input_table_configs["input_1"]["primary_keys"]

# COMMAND ----------

features_data = ft_data.select([input_table_configs["input_1"]["primary_keys"]] + feature_columns)
ground_truth_data = gt_data.select([input_table_configs["input_2"]["primary_keys"]] + target_columns)

# COMMAND ----------

# DBTITLE 1,Joining Feature and Ground truth tables on primary key
final_df = features_data.join(ground_truth_data, on = input_table_configs["input_1"]["primary_keys"])

# COMMAND ----------

# DBTITLE 1,Converting the Spark df to Pandas df
final_df.show()

# COMMAND ----------

print(f"Shape: ({final_df.count()}, {len(final_df.columns)})")

# COMMAND ----------

# DBTITLE 1,Dropping the null rows in the final df
final_df = final_df.dropna()

# COMMAND ----------

print(f"Shape: ({final_df.count()}, {len(final_df.columns)})")

# COMMAND ----------

target_columns

# COMMAND ----------

# DBTITLE 1,Spliting the Final df to test and train dfs
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import col

# Prepare feature and target columns
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
final_df = assembler.transform(final_df)

# Split the data into train and test sets
train_df, test_df = final_df.randomSplit([0.67, 0.33], seed=42)

train_df = train_df.select("features", *target_columns, *feature_columns)
test_df = test_df.select("features", *target_columns, *feature_columns)

# COMMAND ----------

from MLCORE_SDK.helpers.mlc_helper import get_job_id_run_id
job_id, run_id, task_id = get_job_id_run_id(dbutils)
print(job_id, run_id)
report_directory = f'{env}/media_artifacts/2a3b88f5bb6444b0a19e23e4ef21495a/Solution_configs_upgrades/{job_id}/{run_id}/Tuning_Trails'

# COMMAND ----------

# DBTITLE 1,Get Hyper Parameter Tuning Result
try :
    hp_tuning_result = dbutils.notebook.run(
        "Hyperparameter_Tuning", 
        timeout_seconds=0)
    hyperparameters = json.loads(hp_tuning_result)["best_hyperparameters"]
    report_path = json.loads(hp_tuning_result)["report_path"]
except Exception as e:
    print(e)
    hyperparameters = {}
    hp_tuning_result = {}

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline

if not hyperparameters or hyperparameters == {}:
    lr = LogisticRegression(labelCol=target_columns[0], featuresCol="features")
    print("Using model with default hyperparameters")
else:
    lr = LogisticRegression(labelCol=target_columns[0], featuresCol="features", **hyperparameters)
    print("Using model with custom hyperparameters")

pipeline = Pipeline(stages=[lr])

# COMMAND ----------

first_row_dict = train_df.limit(5)

# COMMAND ----------

# DBTITLE 1,Fitting the pipeline on Train data 
# Fit the pipeline
lr_pipeline = pipeline.fit(train_df)

# COMMAND ----------

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Predict on the test set
y_train_pred = lr_pipeline.transform(train_df)
y_pred = lr_pipeline.transform(test_df)

evaluator = MulticlassClassificationEvaluator(labelCol=target_columns[0], predictionCol="prediction")

accuracy = evaluator.evaluate(y_pred, {evaluator.metricName: "accuracy"})
f1 = evaluator.evaluate(y_pred, {evaluator.metricName: "f1"})
precision = evaluator.evaluate(y_pred, {evaluator.metricName: "weightedPrecision"})
recall = evaluator.evaluate(y_pred, {evaluator.metricName: "weightedRecall"})

# COMMAND ----------

y_pred.columns

# COMMAND ----------

# DBTITLE 1,Displaying the test metrics 
test_metrics = {"accuracy":accuracy, "f1":f1, "precision":precision, "recall":recall}
test_metrics

# COMMAND ----------

accuracy = evaluator.evaluate(y_train_pred, {evaluator.metricName: "accuracy"})
f1 = evaluator.evaluate(y_train_pred, {evaluator.metricName: "f1"})
precision = evaluator.evaluate(y_train_pred, {evaluator.metricName: "weightedPrecision"})
recall = evaluator.evaluate(y_train_pred, {evaluator.metricName: "weightedRecall"})

# COMMAND ----------

train_metrics = {"accuracy":accuracy, "f1":f1, "precision":precision, "recall":recall}
train_metrics

# COMMAND ----------

# MAGIC %md
# MAGIC ## SAVE PREDICTIONS TO HIVE

# COMMAND ----------

from pyspark.sql.functions import lit, udf
from pyspark.sql.types import ArrayType, DoubleType
from pyspark.ml.feature import VectorAssembler

# Make predictions
predictions_train = lr_pipeline.transform(train_df)
predictions_test = lr_pipeline.transform(test_df)

# Add dataset type column
predictions_train = predictions_train.withColumn("dataset_type_71E4E76EB8C12230B6F51EA2214BD5FE", lit("train"))
predictions_test = predictions_test.withColumn("dataset_type_71E4E76EB8C12230B6F51EA2214BD5FE", lit("test"))

train_output_df = predictions_train.union(predictions_test)

columns_to_select = feature_columns + target_columns + ["prediction", "dataset_type_71E4E76EB8C12230B6F51EA2214BD5FE", "probability"]
train_output_df = train_output_df.select(columns_to_select)

# COMMAND ----------

train_output_df.columns

# COMMAND ----------

from datetime import datetime
from pyspark.sql import functions as F
from pyspark.sql.window import Window

def to_date_(col):
    """
    Checks col row-wise and returns first date format which returns non-null output for the respective column value
    """
    formats = (
        "MM-dd-yyyy",
        "dd-MM-yyyy",
        "MM/dd/yyyy",
        "yyyy-MM-dd",
        "M/d/yyyy",
        "M/dd/yyyy",
        "MM/dd/yy",
        "MM.dd.yyyy",
        "dd.MM.yyyy",
        "yyyy-MM-dd",
        "yyyy-dd-MM",
    )
    return F.coalesce(*[F.to_date(col, f) for f in formats])

# COMMAND ----------

# DBTITLE 1,Adding Timestamp and Date Features to a Source 1
now = datetime.now()
date = now.strftime("%m-%d-%Y")
train_output_df = train_output_df.withColumn(
    "timestamp",
    F.expr("reflect('java.lang.System', 'currentTimeMillis')").cast("long"),
)
train_output_df = train_output_df.withColumn("date", F.lit(date))
train_output_df = train_output_df.withColumn("date", to_date_(F.col("date")))

# ADD A MONOTONICALLY INREASING COLUMN
if "id" not in train_output_df.columns : 
  window = Window.orderBy(F.monotonically_increasing_id())
  train_output_df = train_output_df.withColumn("id", F.row_number().over(window))

# COMMAND ----------

db_name = output_table_configs["output_1"]["schema"]
table_name = output_table_configs["output_1"]["table"]
catalog_name = output_table_configs["output_1"]["catalog_name"]
output_path = output_table_paths["output_1"]

# Get the catalog name from the table name
if catalog_name and catalog_name.lower() != "none":
  spark.sql(f"USE CATALOG {catalog_name}")


# Create the database if it does not exist
spark.sql(f"CREATE DATABASE IF NOT EXISTS {db_name}")
print(f"HIVE METASTORE DATABASE NAME : {db_name}")

train_output_df.createOrReplaceTempView(table_name)

feature_table_exist = [True for table_data in spark.catalog.listTables(db_name) if table_data.name.lower() == table_name.lower() and not table_data.isTemporary]

if not any(feature_table_exist):
  print(f"CREATING SOURCE TABLE")
  spark.sql(f"CREATE TABLE IF NOT EXISTS {output_path} AS SELECT * FROM {table_name}")
else :
  print(F"UPDATING SOURCE TABLE")
  spark.sql(f"INSERT INTO {output_path} SELECT * FROM {table_name}")

if catalog_name and catalog_name.lower() != "none":
  output_1_table_path = output_path
else:
  output_1_table_path = spark.sql(f"desc formatted {output_path}").filter(F.col("col_name") == "Location").select("data_type").collect()[0][0]

print(f"Features Hive Path : {output_1_table_path}")

# COMMAND ----------

if input_table_configs["input_1"]["catalog_name"]:
    feature_table_path = input_table_paths["input_1"]
else:
    feature_table_path = spark.sql(f"desc formatted {input_table_paths['input_1']}").filter(F.col("col_name") == "Location").select("data_type").collect()[0][0]

if input_table_configs["input_2"]["catalog_name"]:
    gt_table_path = input_table_paths["input_2"]
gt_table_path = spark.sql(f"desc formatted {input_table_paths['input_2']}").filter(F.col("col_name") == "Location").select("data_type").collect()[0][0]

print(feature_table_path, gt_table_path)

# COMMAND ----------

stagemetrics.end()
taskmetrics.end()

stage_Df = stagemetrics.create_stagemetrics_DF("PerfStageMetrics")
task_Df = taskmetrics.create_taskmetrics_DF("PerfTaskMetrics")

compute_metrics = stagemetrics.aggregate_stagemetrics_DF().select("executorCpuTime", "peakExecutionMemory","memoryBytesSpilled","diskBytesSpilled").collect()[0].asDict()

compute_metrics['executorCpuTime'] = compute_metrics['executorCpuTime']/1000
compute_metrics['peakExecutionMemory'] = float(compute_metrics['peakExecutionMemory']) /(1024*1024)

# COMMAND ----------

train_data_date_dict = {
    "feature_table" : {
        "ft_start_date" : ft_data.select(F.min("timestamp")).collect()[0][0],
        "ft_end_date" : ft_data.select(F.max("timestamp")).collect()[0][0]
    },
    "gt_table" : {
        "gt_start_date" : gt_data.select(F.min("timestamp")).collect()[0][0],
        "gt_end_date" : gt_data.select(F.max("timestamp")).collect()[0][0]        
    }
}

# COMMAND ----------

# Calling job run add for DPD job runs
mlclient.log(
    operation_type="job_run_add", 
    session_id = sdk_session_id, 
    dbutils = dbutils, 
    request_type = "train", 
    job_config = 
    {
        "table_name" : output_table_configs["output_1"]["table"],
        "table_type" : "Source",
        "model_name" : model_configs["model_params"]["model_name"],
        "feature_table_path" : feature_table_path,
        "ground_truth_table_path" : gt_table_path,
        "feature_columns" : feature_columns,
        "target_columns" : target_columns,
        "model" : lr_pipeline,
        "model_runtime_env" : "spark",
        "reuse_train_session" : False
    },
    tracking_env = env,
    spark = spark,
    verbose = True,
    )

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## REGISTER MODEL IN MLCORE

# COMMAND ----------

from MLCORE_SDK import mlclient

# COMMAND ----------

# DBTITLE 1,Registering the model in MLCore
model_artifact_id = mlclient.log(operation_type = "register_model",
    sdk_session_id = sdk_session_id,
    dbutils = dbutils,
    spark = spark,
    model = lr_pipeline,
    model_name = model_configs["model_params"]["model_name"],
    model_runtime_env = "spark",
    train_metrics = train_metrics,
    test_metrics = test_metrics,
    feature_table_path = feature_table_path,
    ground_truth_table_path = gt_table_path,
    train_output_path = output_1_table_path,
    train_output_rows = train_output_df.count(),
    train_output_cols = train_output_df.columns,
    table_schema=train_output_df.schema,
    column_datatype = train_output_df.dtypes,
    feature_columns = feature_columns,
    target_columns = target_columns,
    table_type="unitycatalog" if output_table_configs["output_1"]["catalog_name"] else "internal",
    train_data_date_dict = train_data_date_dict,
    hp_tuning_result=hp_tuning_result,
    compute_usage_metrics = compute_metrics,
    taskmetrics = taskmetrics,
    stagemetrics = stagemetrics,
    tracking_env = env,
    model_configs = model_configs,
    example_input = first_row_dict,
    # register_in_feature_store=True,
    verbose = True)

# COMMAND ----------

from MLCORE_SDK.helpers.mlc_helper import get_job_id_run_id
job_id, run_id, task_id = get_job_id_run_id(dbutils)
print(job_id, run_id)

# COMMAND ----------

from MLCORE_SDK.sdk.manage_sdk_session import get_session
existing_session_data = get_session(
    sdk_session_id,
    dbutils,
    api_endpoint=tracking_url,
    tracking_env=tracking_env,
).json()["data"]
existing_session_state = existing_session_data.get("state_dict", {})
project_id = existing_session_state.get("project_id", "")
version = existing_session_state.get("version", "")
print(project_id, version)

# COMMAND ----------

try:
    if storage_configs["cloud_provider"] == "databricks_uc":
        catalog_name = storage_configs.get("catalog_name")
        schema_name = storage_configs.get("schema_name")
        volume_name = storage_configs.get("volume_name")

        artifact_path_uc_volume = f"/Volumes/{catalog_name}/{schema_name}/{volume_name}/{tracking_env}/media_artifacts/{project_id}/{version}/{job_id}/{run_id}"
        print(artifact_path_uc_volume)
        mlclient.log(
            operation_type = "upload_blob_to_cloud",
            blob_path=report_path,
            dbutils = dbutils ,
            target_path = f"{artifact_path_uc_volume}/Model_Evaluation/Tuning_Trails_report_{int(time.time())}.png",
            resource_type = "databricks_uc",
            project_id = project_id,
            version = version,
            job_id = job_id,
            run_id = run_id,
            request_type = "Model_Evaluation",
            storage_configs = storage_configs,
            tracking_env = tracking_env,
            verbose=True)
    else :
        report_directory = f"{tracking_env}/media_artifacts/{project_id}/{version}/{job_id}/{run_id}"
        print(report_directory)
        container_name = storage_configs.get("container_name")
        mlclient.log(
            operation_type = "upload_blob_to_cloud",
            source_path=report_path,
            dbutils = dbutils ,
            target_path = f"{report_directory}/Model_Evaluation/Tuning_Trails_report_{int(time.time())}.png",
            resource_type = "az",
            project_id = project_id,
            version = version,
            job_id = job_id,
            run_id = run_id,
            model_artifact_id = model_artifact_id,
            request_type = "Model_Evaluation",
            storage_configs = storage_configs,
            tracking_env = tracking_env,
            verbose=True)
except Exception as err:
    print(Exception, err)
    
