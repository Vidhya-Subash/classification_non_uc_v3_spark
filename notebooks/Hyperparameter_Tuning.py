# Databricks notebook source
# MAGIC %md ## Hyperparameter_Tuning
# MAGIC

# COMMAND ----------

# %pip install numpy==1.19.1
# %pip install pandas==1.0.5
%pip install kaleido

# COMMAND ----------

# %pip install azure-storage-blob
# %pip install azure-identity

# COMMAND ----------

from sklearn.model_selection import train_test_split
from hyperopt import tpe, fmin, STATUS_OK, Trials, SparkTrials, space_eval
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from sklearn.model_selection import cross_val_score
from hyperopt.pyll import scope
from hyperopt import fmin, tpe, hp, Trials
from hyperopt.spark import SparkTrials
import numpy as np
import json, time
import pandas as pd
import matplotlib.pyplot as plt

# Disable auto-logging of runs in the mlflow
import mlflow
mlflow.autolog(disable=True)

# COMMAND ----------

import yaml
from utils import utils

with open('../data_config/SolutionConfig.yaml', 'r') as solution_config:
    solution_config = yaml.safe_load(solution_config)

# COMMAND ----------

# tracking_env = solution_config["general_configs"]["tracking_env"]
# sdk_session_id = solution_config["general_configs"]["sdk_session_id"][tracking_env]

# JOB SPECIFIC PARAMETERS
input_table_configs = solution_config["train"]["datalake_configs"]["input_tables"]
output_table_configs = solution_config["train"]["datalake_configs"]['output_tables']
model_configs = solution_config["train"]["model_configs"]
feature_columns = solution_config['train']["feature_columns"]
target_columns = solution_config['train']["target_columns"]
test_size = solution_config['train']["test_size"]
primary_keys = input_table_configs["input_2"]["primary_keys"]
primary_metric = solution_config['train']["hyperparameter_tuning"]["primary_metric"]
search_range = solution_config['train']["hyperparameter_tuning"]["search_range"]
max_evaluations = solution_config['train']["hyperparameter_tuning"]["max_evaluations"]
stop_early = solution_config['train']["hyperparameter_tuning"]["stop_early"]
run_parallel = solution_config['train']["hyperparameter_tuning"]["run_parallel"]

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
            table_path = f"hive_metastore.{schema}.{table}"

        data_objects[table_name] = table_path
    
    return data_objects

# COMMAND ----------

input_table_paths = get_name_space(input_table_configs)
output_table_paths = get_name_space(output_table_configs)

# COMMAND ----------

ft_data = spark.sql(f"SELECT * FROM {input_table_paths['input_1']}")
gt_data = spark.sql(f"SELECT * FROM {input_table_paths['input_2']}")

# COMMAND ----------

final_df = ft_data.join(gt_data, on = input_table_configs["input_1"]["primary_keys"])

# COMMAND ----------

final_df.printSchema()

# COMMAND ----------

final_df.show()

# COMMAND ----------

# Split the data
train_df, test_df = final_df.randomSplit([1 - test_size, test_size], seed=0)

# Create a VectorAssembler for feature columns
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")

# COMMAND ----------

def early_stop_function(iteration_stop_count=20, percent_increase=0.0):
        def stop_fn(trials, best_loss=None, iteration_no_progress=0):
            if (
                not trials
                or "loss" not in trials.trials[len(trials.trials) - 1]["result"]
            ):
                return False, [best_loss, iteration_no_progress + 1]
            new_loss = trials.trials[len(trials.trials) - 1]["result"]["loss"]
            if best_loss is None:
                return False, [new_loss, iteration_no_progress + 1]
            best_loss_threshold = best_loss - abs(
                best_loss * (percent_increase / 100.0)
            )
            if new_loss < best_loss_threshold:
                best_loss = new_loss
                iteration_no_progress = 0
            else:
                iteration_no_progress += 1
                print(
                    "No progress made: %d iteration on %d. best_loss=%.2f, best_loss_threshold=%.2f, new_loss=%.2f"
                    % (
                        iteration_no_progress,
                        iteration_stop_count,
                        best_loss,
                        best_loss_threshold,
                        new_loss,
                    )
                )

            return (
                iteration_no_progress >= iteration_stop_count,
                [best_loss, iteration_no_progress],
            )

        return stop_fn

def get_trial_data(trials, search_space):
    if not trials:
        return []

    trial_data = []
    trial_id = 0

    for trial in trials.trials:
        trial_id += 1
        trial["result"]["trial"] = trial_id
        trial["result"]["loss"] = (
            0
            if not np.isnan(trial["result"]["loss"])
            and abs(trial["result"]["loss"]) == np.inf
            else trial["result"]["loss"]
        )

        hp_vals = {}
        for hp, hp_val in trial["misc"]["vals"].items():
            hp_vals[hp] = hp_val[0]

        trial["result"]["hyper_parameters"] = space_eval(
            search_space, hp_vals
        )
        trial_data.append(trial["result"])
    return trial_data

def objective(params):
    start_time = time.time()
    metrics = {}
    lr = LogisticRegression(labelCol=target_columns[0], featuresCol="features", **params)
    model = Pipeline(stages=[assembler, lr])

    model = model.fit(train_df)
    y_test_pred = model.transform(test_df)

    is_multiclass = y_test_pred.select(F.countDistinct("prediction")).first()[0] > 2
    metrics_average = "weighted" if is_multiclass else "binary"

    # Calculate metrics
    evaluator = MulticlassClassificationEvaluator(labelCol=target_columns[0], predictionCol="prediction")

    metrics = {}
    metrics["f1"] = evaluator.evaluate(y_test_pred, {evaluator.metricName: "f1"})
    metrics["precision"] = evaluator.evaluate(y_test_pred, {evaluator.metricName: "weightedPrecision"})
    metrics["accuracy"] = evaluator.evaluate(y_test_pred, {evaluator.metricName: "accuracy"})
    metrics["recall"] = evaluator.evaluate(y_test_pred, {evaluator.metricName: "weightedRecall"})
    
    loss = - metrics[primary_metric]
    end_time = time.time()
    
    trail_output = {
        "loss": abs(loss),
        "metrics": metrics,
        "status": STATUS_OK,
        "duration" : end_time - start_time,
        "primary_metric":primary_metric,
        "max_evaluations":max_evaluations,
        "early_stopping":stop_early}

    return trail_output

def hyperparameter_tuning_with_trials(search_space,max_evals,run_parallel,stop_early):
    if run_parallel:
        trials = SparkTrials(parallelism=4)
    else:
        trials = Trials()

    best_config = fmin(
            fn=objective,
            space=search_space,
            algo=tpe.suggest,
            max_evals= max_evals,
            trials=trials,
            early_stop_fn= early_stop_function(10, -0.01)
            if stop_early
            else None,
        )

    return best_config, trials


hyperopt_mapping = {
    bool: hp.choice,
    int:  hp.quniform,
    float: hp.uniform,
    str: hp.choice
}

# Converted search space
search_space = {}

for key, value in search_range.items():
    value_type = type(value[0])
    if value_type in hyperopt_mapping:
        if value_type in [bool, str]:
            search_space[key] = hyperopt_mapping[value_type](key, value)
        elif value_type is int:
            search_space[key] = scope.int(hyperopt_mapping[value_type](key, value[0], value[1], 1))
        else:
            search_space[key] = hyperopt_mapping[value_type](key, value[0], value[1])
    else:
        raise ValueError(f"Unsupported type for {key}")

# COMMAND ----------

best_hyperparameters , tuning_trails = hyperparameter_tuning_with_trials( search_space= search_space, max_evals=max_evaluations, run_parallel=run_parallel,stop_early=stop_early)

best_hyperparameters = space_eval(search_space, best_hyperparameters)
tuning_trails_all = get_trial_data(tuning_trails, search_space)


# COMMAND ----------

tuning_trails_all

# COMMAND ----------

df = pd.json_normalize(tuning_trails_all)

# COMMAND ----------

df.display()

# COMMAND ----------

import plotly.express as px
import plotly.io as pio

# COMMAND ----------

fig = px.bar(df, x='trial', y='loss', title='Trial vs Loss', labels={'trial': 'Trial', 'loss': 'Loss'}, color_discrete_sequence=['blue'])

# COMMAND ----------

fig

# COMMAND ----------

report_path = f"/dbfs/FileStore/Amplify/Tuning_Trails_report_{int(time.time())}.png"
pio.write_image(fig, report_path)
print(report_path)

# COMMAND ----------

hp_tuning_result = {
    "best_hyperparameters":best_hyperparameters,
    "tuning_trails":tuning_trails_all,
    "report_path": report_path,
}

# COMMAND ----------

hp_tuning_result

# COMMAND ----------

dbutils.notebook.exit(json.dumps(hp_tuning_result))
